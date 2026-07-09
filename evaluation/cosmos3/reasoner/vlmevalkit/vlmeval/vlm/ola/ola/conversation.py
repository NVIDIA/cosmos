import base64
import dataclasses
from enum import Enum, auto
from io import BytesIO
from typing import Any, List, Tuple, Union

from PIL import Image


class SeparatorStyle(Enum):
    """Different separator style."""
    TWO = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.PLAIN
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.QWEN2:
            start = '<|im_start|>'
            end = '<|im_end|>\n'
            ret = start + 'system\n' + self.system + end
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message

                    if message.endswith('<|endoftext|>'):
                        message = message.replace('<|endoftext|>', '')
                        ret += start + role + "\n" + message + end + '<|endoftext|>'
                    else:
                        assert not '<|endoftext|>' in message, f"Invalid message: {message}"
                        ret += start + role + "\n" + message + end
                else:
                    ret += start + role + "\n"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, speech = msg
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_qwen_v1 = Conversation(
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN2,
)

default_conversation = conv_qwen_v1
conv_templates = {
    'v1_qwen2': conv_qwen_v1,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
