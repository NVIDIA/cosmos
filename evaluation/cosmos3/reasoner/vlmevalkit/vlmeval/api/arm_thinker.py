import base64
import datetime as dt
import json
import logging
import os
import os.path as osp
import sys
import traceback
import uuid

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.smp import encode_image_to_base64


class LMDeployWrapper(BaseAPI):

    is_api: bool = True

    custom_prompt: str = None

    def __init__(
        self,
        model: str = None,
        retry: int = 5,
        api_base: str = None,
        key: str = "sk-123456",
        verbose: bool = True,
        timeout: int = 60,
        agent_repo_root: str = None,
        use_role_tool: bool = True,
        **kwargs,
    ):
        # get key and api_base from environment variables
        key = os.environ.get("LMDEPLOY_API_KEY", key)
        api_base = os.environ.get("LMDEPLOY_API_BASE", api_base)
        assert key is not None, "Please set the environment variable LMDEPLOY_API_KEY."
        assert (
            api_base is not None
        ), "Please set the environment variable LMDEPLOY_API_BASE."
        self.key = key
        self.api_base = api_base
        model_url = "".join([api_base.split("v1")[0], "v1/models"])
        headers = {"Authorization": f"Bearer {self.key}"}
        resp = requests.get(model_url, headers=headers)
        model_id_list = [str(data["id"]) for data in resp.json()["data"]]
        self.model = model if model in model_id_list else model_id_list[0]

        self.fail_msg = "Failed to obtain answer via API. "
        self.timeout = timeout
        self.extra_pt = kwargs.pop("extra_pt", None)
        self.use_role_tool = use_role_tool
        # Set up logging for agent mode if needed
        self.debug_mode = kwargs.pop("debug_mode", False)
        if self.debug_mode:
            logging.basicConfig(level=logging.DEBUG, force=True)

        super().__init__(
            retry=retry, verbose=verbose, **kwargs
        )

        if agent_repo_root is None:
            raise ValueError('Please set `agent_repo_root` to ARM-Thinker directory, \
                          which is cloned from here: https://github.com/InternLM/ARM-Thinker')

        print(f"agent_repo_root: {agent_repo_root}")
        sys.path.append(agent_repo_root)
        try:
            from arm_agent.agent_verl import VerlAgent  # noqa: F401
        except Exception:
            raise ValueError('Please install ARM-Thinker from https://github.com/InternLM/ARM-Thinker')

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg["type"] == "text":
                    content_list.append(dict(type="text", text=msg["value"]))
                elif msg["type"] == "image":
                    from PIL import Image

                    img = Image.open(msg["value"])
                    b64 = encode_image_to_base64(img)
                    extra_args = msg.copy()
                    extra_args.pop("type")
                    extra_args.pop("value")
                    img_struct = dict(url=f"data:image/jpeg;base64,{b64}", **extra_args)
                    content_list.append(dict(type="image_url", image_url=img_struct))
        else:
            assert all([x["type"] == "text" for x in inputs])
            text = "\n".join([x["value"] for x in inputs])
            content_list = [dict(type="text", text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(["type" in x for x in inputs]) or np.all(
            ["role" in x for x in inputs]
        ), inputs
        if "role" in inputs[0]:
            assert inputs[-1]["role"] == "user", inputs[-1]
            for item in inputs:
                input_msgs.append(
                    dict(role=item["role"], content=self.prepare_itlist(item["content"]))
                )
        else:
            input_msgs.append(dict(role="user", content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> tuple:
        mode = kwargs.pop("mode", "direct")
        dataset = kwargs.pop("dataset", None)

        if mode == "direct":
            input_msgs = self.prepare_inputs(inputs)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            }

            if self.extra_pt:
                input_msgs[-1]["content"][-1]["text"] = (
                    input_msgs[-1]["content"][-1]["text"] + self.extra_pt
                )

            payload = {
                "model": self.model,
                "messages": input_msgs,
                "n": 1,
                "extra_body": {},
            }
            if kwargs.get("repetition_penalty", None):
                payload["extra_body"]["repetition_penalty"] = kwargs.get("repetition_penalty")
                kwargs.pop("repetition_penalty")

            if kwargs.get("top_k", None):
                payload["extra_body"]["top_k"] = kwargs.get("top_k")
                kwargs.pop("top_k")
            payload.update(kwargs)

            payload_copy = payload.copy()
            payload_copy.pop("messages")
            print(f"Full payload except messages:\n{payload_copy}")

            response = requests.post(
                self.api_base,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout * 1.1,
            )
            ret_code = response.status_code
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
            answer = self.fail_msg
            try:
                resp_struct = json.loads(response.text)
                answer = resp_struct["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
            return ret_code, answer, response

        elif mode == "agent":
            timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            temp_dir = f"/tmp/agent_images_{timestamp}"
            os.environ["TOOL_CALL_IMG_TEMP"] = temp_dir
            os.makedirs(temp_dir, exist_ok=True)

            from arm_agent.agent_verl import VerlAgent
            agent = VerlAgent(
                model_name=self.model,
                api_base=self.api_base.split("chat/completions")[0],
                api_key=self.key,
                use_role_tool=self.use_role_tool,
                **kwargs
            )

            input_msgs = self.prepare_inputs(inputs)

            if self.extra_pt:
                input_msgs[-1]["content"][-1]["text"] = (
                    input_msgs[-1]["content"][-1]["text"] + self.extra_pt
                )

            result, tool_call_count = agent.run(user_messages=input_msgs)
            rtn = result[-1]["content"]

            # "base64" or "save_to_path" or "hidden", default is "base64"
            # Option "base64": Direct save the image base64 to result file, which will be large.
            # Option "save_to_path": Save the image to path, which will be large.
            # Option "hidden": Hidden the image, which will be small.
            option_for_process_image_in_result = "save_to_path"
            if option_for_process_image_in_result == "hidden":
                for msg in result:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    # print image part, base64 is too long to print
                    if role == "user" and isinstance(content, list):
                        for item in content:
                            if "type" in item and item["type"] == "image_url":
                                item["image_url"]["url"] = "hidden"
                    print(f"{role.upper()}:\n{content}\n")
            elif option_for_process_image_in_result == "base64":
                pass
            elif option_for_process_image_in_result == "save_to_path":
                # vlmevalkit_root is path like xxx/VLMEvalKit
                vlmevalkit_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
                save_img_root = osp.join(vlmevalkit_root, "temp", dataset, self.model)
                os.makedirs(save_img_root, exist_ok=True)

                # uuid
                short_uid = uuid.uuid4().hex[:6]  # 6 bit uuid
                sub_dir_name = f"{timestamp}_{short_uid}"
                img_counter = 0
                for msg in result:
                    content = msg.get("content", "")
                    if not isinstance(content, list):
                        continue
                    for itemm in content:
                        if not (isinstance(itemm, dict) and itemm.get("type") == "image_url"):
                            continue
                        image_url = itemm.get("image_url", {}).get("url", "")
                        if not image_url:
                            continue
                        try:
                            # judge whether it is base64 or url
                            if image_url.startswith("data:image"):
                                header, b64_data = image_url.split(",", 1)
                                ext = header.split("/")[1].split(";")[0]  # extract extension
                                img_bytes = base64.b64decode(b64_data)
                                img_name = f"{sub_dir_name}/img_{img_counter:04d}.{ext}"
                            else:
                                raise ValueError(
                                    f"Invalid image url: {image_url}, can only process base64 image url"
                                )
                            # save image
                            img_path = os.path.join(save_img_root, img_name)
                            os.makedirs(os.path.dirname(img_path), exist_ok=True)
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)

                            # replace image_url["url"] with relative path
                            itemm["image_url"]["url"] = f"{img_name}"
                            img_counter += 1

                            print(f"[Saved] {img_path}, {itemm['image_url']['url']}")
                        except Exception as e:
                            print(f"Error: {e}")
                            print("Traceback:\n" + traceback.format_exc())
                            raise ValueError(f"Error: {e}")
            else:
                raise ValueError(f"Invalid option: {option_for_process_image_in_result}")

            extra_records = {"tool_call_count": tool_call_count, "conversation": result}
            print(f"tool_call_count in extra_records: {tool_call_count}")

            if rtn:
                ret_code = 0
                return ret_code, rtn, extra_records
            else:
                ret_code = 1
                return ret_code, self.fail_msg, extra_records
        else:
            raise ValueError(f"Invalid mode: {mode}")


class ARM_thinker(LMDeployWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(ARM_thinker, self).generate(message, dataset=dataset)
