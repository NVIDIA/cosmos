import copy as cp
import os
from functools import partial

import vlmeval.api as api
import vlmeval.vlm as vlm

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = "Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. "

video_models = {
    "Cosmos-Reason1-7B": partial(
        vlm.CosmosReasonHF,
        model_path="nvidia/Cosmos-Reason1-7B",
    ),
    # An example of a finetuned Reason1 model. Model path can be on the HuggingFace hub or locally.
    "Cosmos-Reason1-7B-VideoPhy2": partial(
        vlm.CosmosReasonHF,
        model_path="shunznv/Cosmos-Reason1-7B-VideoPhy2",
    ),
    "Video-LLaVA-7B": partial(vlm.VideoLLaVA, model_path="LanguageBind/Video-LLaVA-7B"),
    "Video-LLaVA-7B-HF": partial(
        vlm.VideoLLaVA_HF, model_path="LanguageBind/Video-LLaVA-7B-hf"
    ),
    "VideoChat2-HD": partial(
        vlm.VideoChat2_HD,
        model_path="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B",
        root=VideoChat2_ROOT,
        config_file="./vlmeval/vlm/video_llm/configs/videochat2_hd.json",
    ),
    "Chat-UniVi-7B": partial(vlm.Chatunivi, model_path="Chat-UniVi/Chat-UniVi"),
    "Chat-UniVi-7B-v1.5": partial(
        vlm.Chatunivi, model_path="Chat-UniVi/Chat-UniVi-7B-v1.5"
    ),
    "LLaMA-VID-7B": partial(
        vlm.LLaMAVID, model_path="YanweiLi/llama-vid-7b-full-224-video-fps-1"
    ),
    "Video-ChatGPT": partial(
        vlm.VideoChatGPT, model_path="MBZUAI/Video-ChatGPT-7B", dir_root=VideoChatGPT_ROOT
    ),
    "PLLaVA-7B": partial(vlm.PLLaVA, model_path="ermu2001/pllava-7b", dir_root=PLLaVA_ROOT),
    "PLLaVA-13B": partial(
        vlm.PLLaVA, model_path="ermu2001/pllava-13b", dir_root=PLLaVA_ROOT
    ),
    "PLLaVA-34B": partial(
        vlm.PLLaVA, model_path="ermu2001/pllava-34b", dir_root=PLLaVA_ROOT
    ),
}

ungrouped = {
    "AKI": partial(vlm.AKI, name="AKI", ckpt_pth="Sony/AKI-4B-phi-3.5-mini"),
    "TransCore_M": partial(vlm.TransCoreM, root=TransCore_ROOT),
    "PandaGPT_13B": partial(vlm.PandaGPT, name="PandaGPT_13B", root=PandaGPT_ROOT),
    "flamingov2": partial(
        vlm.OpenFlamingo,
        name="v2",
        mpt_pth="anas-awadalla/mpt-7b",
        ckpt_pth="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    ),
    "VisualGLM_6b": partial(vlm.VisualGLM, model_path="THUDM/visualglm-6b"),
    "mPLUG-Owl2": partial(vlm.mPLUG_Owl2, model_path="MAGAer13/mplug-owl2-llama2-7b"),
    "mPLUG-Owl3": partial(vlm.mPLUG_Owl3, model_path="mPLUG/mPLUG-Owl3-7B-240728"),
    "OmniLMM_12B": partial(
        vlm.OmniLMM12B, model_path="openbmb/OmniLMM-12B", root=OmniLMM_ROOT
    ),
    "MGM_7B": partial(
        vlm.Mini_Gemini, model_path="YanweiLi/MGM-7B-HD", root=Mini_Gemini_ROOT
    ),
    "Bunny-llama3-8B": partial(vlm.BunnyLLama3, model_path="BAAI/Bunny-v1_1-Llama-3-8B-V"),
    "VXVERSE": partial(vlm.VXVERSE, model_name="XVERSE-V-13B", root=VXVERSE_ROOT),
    "360VL-70B": partial(vlm.QH_360VL, model_path="qihoo360/360VL-70B"),
    "Llama-3-MixSenseV1_1": partial(
        vlm.LLama3Mixsense, model_path="Zero-Vision/Llama-3-MixSenseV1_1"
    ),
    "Parrot": partial(vlm.Parrot, model_path="AIDC-AI/Parrot-7B"),
    "OmChat": partial(vlm.OmChat, model_path="omlab/omchat-v2.0-13B-single-beta_hf"),
    "RBDash_72b": partial(
        vlm.RBDash, model_path="RBDash-Team/RBDash-v1.5", root=RBDash_ROOT
    ),
    "Pixtral-12B": partial(vlm.Pixtral, model_path="mistralai/Pixtral-12B-2409"),
    "Ministral-3-14B-Instruct-2512_api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='Ministral-3-14B-Instruct-2512',
        temperature=0.15,
        max_new_tokens=32768,
        retry=6,
        timeout=1800,
    ),
    "Falcon2-VLM-11B": partial(vlm.Falcon2VLM, model_path="tiiuae/falcon-11B-vlm"),
    "KVL": partial(vlm.InternVLChat, model_path="amoeba04/KVL", version="V2.0"),
}

o1_key = os.environ.get('O1_API_KEY', None)
o1_base = os.environ.get('O1_API_BASE', None)
o1_apis = {
    'o1': partial(
        api.GPT4V,
        model="o1-2024-12-17",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800, 
        max_tokens=16384,
        verbose=False,

    ),
    'o3': partial(
        api.GPT4V, 
        model="o3-2025-04-16",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800, 
        max_tokens=16384, 
        verbose=False,
    ),
    'o4-mini': partial(
        api.GPT4V, 
        model="o4-mini-2025-04-16",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
}

api_models = {
    # Lepton API models
    "qwen3_235b_a22b": partial(
        api.CosmosReason2,
        model="Qwen3-VL-235B-A22B-Instruct",
        api_base="https://b5k2m9x7-qwen3-vl-235b-a22b-instruct.xenon.lepton.run/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=300,  # Increased timeout for video processing (5 minutes)
        verbose=False,
    ),
    "qwen3_30b_a3b": partial(
        api.CosmosReason2,
        model="Qwen3-VL-30B-A3B-Instruct",
        api_base="https://b5k2m9x7-qwen3-vl-30b-a3b-instruct.xenon.lepton.run/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=300,  # Increased timeout for video processing (5 minutes)
    ),
    "qwen3_8b": partial(
        api.CosmosReason2,
        model="Qwen3-VL-8B-Instruct",
        api_base="https://b5k2m9x7-qwen3-vl-8b.xenon.lepton.run/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "cosmos_reason1_7b": partial(
        api.CosmosReason1,
        model="Cosmos-Reason1-7B",
        api_base="https://b5k2m9x7-cosmos-reason1-7b-1107.xenon.lepton.run/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "GeminiPro3-0": partial(
        api.Gemini, model="gemini-3-pro-preview", temperature=0, retry=10, max_tokens=16384
    ),
    "LVS_service": partial(
        api.LVS_service,
        model="Qwen/Qwen3-VL-8B-Instruct",
        temperature=0,
        retry=5,
        verbose=False,
        max_tokens=16384,
        api_base="http://10.111.53.95:8009",
        fail_msg={"events": [], "total_events": 0, "video_summary": ""},
        chunk_duration=10,
        num_frames_per_chunk=15,
    ),
    # GPT
    "GPT4V": partial(
        api.GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_HIGH": partial(
        api.GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409": partial(
        api.GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409_HIGH": partial(
        api.GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o": partial(
        api.GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4o_HIGH": partial(
        api.GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20240806": partial(
        api.GPT4V,
        model="gpt-4o-2024-08-06",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20241120": partial(
        api.GPT4V,
        model="gpt-4o-2024-11-20",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "ChatGPT4o": partial(
        api.GPT4V,
        model="chatgpt-4o-latest",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_MINI": partial(
        api.GPT4V,
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4.5": partial(
        api.GPT4V, 
        model='gpt-4.5-preview-2025-02-27',
        temperature=0, 
        timeout=600,
        img_size=-1, 
        img_detail='high', 
        retry=10, 
        verbose=False,
    ),
    "gpt-4.1-2025-04-14": partial(
        api.GPT4V,
        model="gpt-4.1-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-mini-2025-04-14": partial(
        api.GPT4V,
        model="gpt-4.1-mini-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-nano-2025-04-14": partial(
        api.GPT4V,
        model="gpt-4.1-nano-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-5-2025-08-07": partial(
        api.GPT4V,
        model="gpt-5-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=3000,
    ),
    "gpt-5-mini-2025-08-07": partial(
        api.GPT4V,
        model="gpt-5-mini-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5-nano-2025-08-07": partial(
        api.GPT4V,
        model="gpt-5-nano-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5.1-2025-11-13": partial(
        api.GPT4V,
        model="gpt-5.1-2025-11-13",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    # Gemini
    "GeminiPro1-0": partial(
        api.Gemini, model="gemini-1.0-pro", temperature=0, retry=10
    ),  # now GeminiPro1-0 is only supported by vertex backend
    "GeminiPro1-5": partial(
        api.Gemini, model="gemini-1.5-pro", temperature=0, retry=10
    ),
    "GeminiFlash1-5": partial(
        api.Gemini, model="gemini-1.5-flash", temperature=0, retry=10
    ),
    "GeminiPro1-5-002": partial(
        api.GPT4V, model="gemini-1.5-pro-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash1-5-002": partial(
        api.GPT4V, model="gemini-1.5-flash-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash2-0": partial(
        api.Gemini, model="gemini-2.0-flash", temperature=0, retry=10
    ),
    "GeminiFlashLite2-0": partial(
        api.Gemini, model="gemini-2.0-flash-lite", temperature=0, retry=10
    ),
    "GeminiFlash2-5": partial(
        api.Gemini, model="gemini-2.5-flash", temperature=0, retry=10
    ),
    "GeminiPro2-5": partial(
        api.GPT4V,
        model="gemini-2.5-pro",
        temperature=0,
        retry=10,
        timeout=6000,
        max_tokens=65536,
    ),
    "Gemini-3.1-Pro-Preview": partial(
        api.GPT4V,
        model="gemini-3.1-pro-preview-thinking",
        retry=10,
        timeout=3600,
        max_tokens=65536,
        img_detail='high',
    ),
    
    # GCP Vertex AI – Claude (same GCPVertexAPI; model name selects Claude backend)
    "GCP_Claude3-5Sonnet": partial(
        api.GCPVertexAPI,
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        retry=10,
    ),
    "GCP_Claude3-5Haiku": partial(
        api.GCPVertexAPI,
        model="claude-3-5-haiku@20241022",
        temperature=0,
        retry=10,
    ),
    "GCP_Claude3-7Sonnet": partial(
        api.GCPVertexAPI,
        model="claude-3-7-sonnet@20250219",
        temperature=0,
        retry=10,
    ),
    "GCP_ClaudeSonnet4-5": partial(
        api.GCPVertexAPI,
        model="claude-sonnet-4-5@20250929",
        temperature=0,
        retry=10,
    ),
    "GCP_ClaudeOpus4-6": partial(
        api.GCPVertexAPI,
        model="claude-opus-4-6",
        temperature=0,
        retry=10,
    ),
    # Qwen-VL
    "QwenVLPlus": partial(api.QwenVLAPI, model="qwen-vl-plus", temperature=0, retry=10),
    "QwenVLMax": partial(api.QwenVLAPI, model="qwen-vl-max", temperature=0, retry=10),
    "QwenVLMax-250408": partial(api.QwenVLAPI, model="qwen-vl-max-2025-04-08", temperature=0, retry=10),

    # Reka
    "RekaEdge": partial(api.Reka, model="reka-edge-20240208"),
    "RekaFlash": partial(api.Reka, model="reka-flash-20240226"),
    "RekaCore": partial(api.Reka, model="reka-core-20240415"),
    # Step1V
    "Step1V": partial(
        api.GPT4V,
        model="step-1v-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1.5V-mini": partial(
        api.GPT4V,
        model="step-1.5v-mini",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1o": partial(
        api.GPT4V,
        model="step-1o-vision-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step3-VL-10B_api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='Step3-VL-10B',
        temperature=1.0, 
        max_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20,
        retry=6,
        timeout=1800,
    ),
    # Yi-Vision
    "Yi-Vision": partial(
        api.GPT4V,
        model="yi-vision",
        api_base="https://api.lingyiwanwu.com/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # Together AI (set TOGETHER_API_KEY)
    "Together_Llama3.2-11B-Vision": partial(
        api.TogetherAPI,
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    "Together_Llama3.2-90B-Vision": partial(
        api.TogetherAPI,
        model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    "Together_Llama4-Scout-17B": partial(
        api.TogetherAPI,
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    "Together_Llama4-Maverick-17B": partial(
        api.TogetherAPI,
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    "Together_Qwen2-VL-72B": partial(
        api.TogetherAPI,
        model="Qwen/Qwen2-VL-72B-Instruct",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    # MiniMax (set MINIMAX_API_KEY)
    "MiniMax-M2.7": partial(
        api.MiniMaxAPI,
        model="MiniMax-M2.7",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    "MiniMax-M2.5": partial(
        api.MiniMaxAPI,
        model="MiniMax-M2.5",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    "MiniMax-M2.5-highspeed": partial(
        api.MiniMaxAPI,
        model="MiniMax-M2.5-highspeed",
        temperature=0,
        max_tokens=2048,
        retry=10,
    ),
    # Claude
    "Claude3V_Opus": partial(
        api.Claude3V, model="claude-3-opus-20240229", temperature=0, retry=10, verbose=False
    ),
    "Claude3V_Sonnet": partial(
        api.Claude3V,
        model="claude-3-sonnet-20240229",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3V_Haiku": partial(
        api.Claude3V,
        model="claude-3-haiku-20240307",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet": partial(
        api.Claude3V,
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet_20241022": partial(
        api.Claude3V,
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-7V_Sonnet": partial(
        api.Claude3V,
        model="claude-3-7-sonnet-20250219",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude4_Opus": partial(
        api.Claude3V,
        model="claude-4-opus-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    "Claude4_Sonnet": partial(
        api.Claude3V,
        model="claude-4-sonnet-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    "Claude-Opus-4-6": partial(
        api.GPT4V,
        model="claude-opus-4-6-thinking",
        retry=10,
        timeout=3600,
        max_tokens=65536,
        img_detail='high',
    ),
    "GPT-5.4-2026-03-05": partial(
        api.GPT4V,
        model="gpt-5.4-2026-03-05",
        retry=10,
        timeout=3600,
        max_tokens=65536,
        img_detail='high',
    ),
    "GPT-5.2": partial(
        api.GPT4V,
        model="gpt-5.2",
        retry=10,
        timeout=3600,
        max_tokens=65536,
        img_detail='high',
    ),
    # AWS Bedrock (Converse API; set AWS_REGION or pass region_name)
    "Bedrock_Claude3-5Sonnet": partial(
        api.BedrockAPI,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0,
        retry=10,
    ),
    "Bedrock_Claude3Opus": partial(
        api.BedrockAPI,
        model_id="anthropic.claude-3-opus-20240229-v1:0",
        temperature=0,
        retry=10,
    ),
    "Bedrock_Claude3Sonnet": partial(
        api.BedrockAPI,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0,
        retry=10,
    ),
    "Bedrock_Claude3Haiku": partial(
        api.BedrockAPI,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0,
        retry=10,
    ),
    # GLM4V
    "GLM4V": partial(api.GLMVisionAPI, model="glm4v-biz-eval", temperature=0, retry=10),
    "GLM4V_PLUS": partial(api.GLMVisionAPI, model="glm-4v-plus", temperature=0, retry=10),
    "GLM4V_PLUS_20250111": partial(
        api.GLMVisionAPI, model="glm-4v-plus-0111", temperature=0, retry=10
    ),
    # MiniMax abab
    "abab6.5s": partial(
        api.GPT4V,
        model="abab6.5s-chat",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "abab7-preview": partial(
        api.GPT4V,
        model="abab7-chat-preview",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # CongRong
    "CongRong-v1.5": partial(api.CWWrapper, model="cw-congrong-v1.5", temperature=0, retry=10),
    "CongRong-v2.0": partial(api.CWWrapper, model="cw-congrong-v2.0", temperature=0, retry=10),
    # SenseNova
    "SenseNova-V6-Pro": partial(
        api.SenseChatVisionAPI, model="SenseNova-V6-Pro", temperature=0, retry=10
    ),
    "SenseNova-V6-Reasoner": partial(
        api.SenseChatVisionAPI, model="SenseNova-V6-Reasoner", temperature=0, retry=10
    ),
    "SenseNova-V6-5-Pro": partial(
        api.SenseChatVisionAPI, model="SenseNova-V6-5-Pro", retry=10
    ),
    "SenseNova-V6-5-Pro-20251215": partial(
        api.SenseChatVisionV2API, model="SenseNova-V6-5-Pro-20251215", max_completion_tokens=40960, repetition_penalty=1.05, temperature=0.6, top_p=0.95, top_k=20, timeout=1800, retry=3, img_size=4096
    ),
    "HunYuan-Vision": partial(
        api.HunyuanVision, model="hunyuan-vision", temperature=0, retry=10
    ),
    "HunYuan-Standard-Vision": partial(
        api.HunyuanVision, model="hunyuan-standard-vision", temperature=0, retry=10
    ),
    "HunYuan-Large-Vision": partial(
        api.HunyuanVision, model="hunyuan-large-vision", temperature=0, retry=10
    ),
    "BailingMM-Lite-1203": partial(
        api.bailingMMAPI, model="BailingMM-Lite-1203", temperature=0, retry=10
    ),
    "BailingMM-Pro-0120": partial(
        api.bailingMMAPI, model="BailingMM-Pro-0120", temperature=0, retry=10
    ),
    # BlueLM-2.5
    "BlueLM-2.5-3B": partial(api.BlueLM_API, model="BlueLM-2.5-3B", temperature=0, retry=3),
    # JiuTian-VL
    "JTVL": partial(api.JTVLChatAPI, model="jt-vl-chat", temperature=0, retry=10),
    "JTVL-Mini": partial(api.JTVLChatAPI_Mini, model="jt-vl-chat-mini", temperature=0, retry=10),
    "JTVL-2B": partial(api.JTVLChatAPI_2B, model="jt-vl-chat-2b", temperature=0, retry=10),
    "VideoChatOnlineV2": partial(api.VideoChatOnlineV2API, model="videochatonline_v2", temperature=0, retry=10),
    "Taiyi": partial(api.TaiyiAPI, model="taiyi", temperature=0, retry=10),
    # TeleMM
    "TeleMM": partial(api.TeleMMAPI, model="TeleAI/TeleMM", temperature=0, retry=10),
    "TeleMM2.0": partial(api.TeleMM2_API, model="TeleAI/TeleMM", retry=3, timeout=600),
    "TeleMM2.0Thinking": partial(api.TeleMM2Thinking_API, model="TeleAI/TeleMM", retry=3, timeout=600),
    "Qwen2.5-VL-32B-Instruct-SiliconFlow": partial(
        api.SiliconFlowAPI, model="Qwen/Qwen2.5-VL-32B-Instruct", temperature=0, retry=10),
    "Qwen3-VL-8B--crop--arm_thinker_prompt--sglang": partial(
        api.ARM_thinker,
        mode="agent",
        agent_repo_root="/path/to/your/ARM-Thinker",
        model="Qwen/Qwen3-VL-8B-Instruct",
        retry=10,
        timeout=300,
        api_base="http://100.97.158.184:38888/v1/chat/completions",
        key="EMPTY",
        temperature=0.0,
        max_tokens=4096,
        # agent params
        max_round=16,
        max_tool_response_length=4096,
        tool_config_path="/path/to/your/ARM-Thinker/examples/self/multiturn/config/tool_config/image_zoom_in_tool_config.yaml",
        # special for sglang server
        use_role_tool=False,
        system_template_type="CommonSystemTemplate",
        # extra prompt to adapt to the ARM-Thinker prompt template [CommonSystemTemplate]
        extra_pt="\n\n**Important Requirement:**\nThe given image is `original_image`. You must output your reasoning inside `<think>...</think>`. After reasoning, either output the final answer within `<answer>...</answer>` or call a tool within `<tool_call>...</tool_call>`. You may call tools multiple times across turns to assist with judgment or verification, **but only one tool per turn**. If a tool call fails, you can retry or stop and give your final answer. Once no more tool calls are needed, provide your final answer or judgment within `<answer>...</answer>`.",
    ),
    "Qwen3-VL-8B--crop--official_prompt--vllm": partial(
        api.ARM_thinker,
        mode="agent",
        agent_repo_root="/path/to/your/ARM-Thinker",
        model="Qwen/Qwen3-VL-8B-Instruct",
        retry=10,
        timeout=300,
        api_base="http://100.97.203.103:40001/v1/chat/completions",
        key="EMPTY",
        temperature=0.0,
        max_tokens=4096,
        extra_pt="",
        # agent params
        max_round=16,
        max_tool_response_length=4096,
        system_template_type="Qwen3VLSystemTemplateWithTools",
        tool_config_path="/path/to/your/ARM-Thinker/examples/self/multiturn/config/tool_config/image_zoom_in_tool_qwen3vl_config.yaml",
        use_role_tool=True,
    ),
    # lmdeploy api
    "lmdeploy_internvl_78B_MPO": partial(
        api.LMDeployAPI,
        model="InternVL2_5-78B-MPO",
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=100,
    ),
    "lmdeploy_qvq_72B_preview": partial(
        api.LMDeployAPI,
        model="QVQ-72B-Preview",
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'Taichu-VLR-3B': partial(
        api.TaichuVLRAPI, 
        model='taichu_vlr_3b', 
        url="https://platform.wair.ac.cn/maas/v1/chat/completions"
    ),
    'Taichu-VLR-7B': partial(
        api.TaichuVLRAPI, 
        model='taichu_vlr_7b', 
        url="https://platform.wair.ac.cn/maas/v1/chat/completions"
    ),
    # doubao_vl
    "DoubaoVL": partial(
        api.DoubaoVL, model="Doubao-1.5-vision-pro", temperature=0, retry=3, verbose=False
    ),
    "Seed1.5-VL": partial(
        api.DoubaoVL, 
        model="doubao-1-5-thinking-vision-pro-250428", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6": partial(
        api.DoubaoVL, 
        model="doubao-seed-1.6-250615", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6-Flash": partial(
        api.DoubaoVL, 
        model="doubao-seed-1.6-flash-250615", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6-Thinking": partial(
        api.DoubaoVL, 
        model="doubao-seed-1.6-thinking-250615", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Doubao-Seed-2.0-Pro-260215": partial(
        api.GPT4V,
        model="doubao-seed-2-0-pro-260215",
        retry=3,
        timeout=1200, 
        max_tokens=32768,
    ),
    # Shopee MUG-U
    'MUG-U-7B': partial(
        api.MUGUAPI, 
        model='MUG-U', 
        temperature=0,  
        retry=10, 
        verbose=False, 
        timeout=300),
    # grok
    "grok-vision-beta": partial(
        api.GPT4V,
        model="grok-vision-beta",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-2-vision-1212": partial(
        api.GPT4V,
        model="grok-2-vision",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-4-0709": partial(
        api.GPT4V,
        model="grok-4-0709",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=1200, 
        max_tokens=16384
    ),
    "Grok-4.1-Fast": partial(
        api.GPT4V,
        model="grok-4-1-fast-reasoning",
        retry=3,
        timeout=1200, 
        max_tokens=16384,
    ),
    # kimi
    "moonshot-v1-8k": partial(
        api.GPT4V,
        model="moonshot-v1-8k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-32k": partial(
        api.GPT4V,
        model="moonshot-v1-32k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-128k": partial(
        api.GPT4V,
        model="moonshot-v1-128k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    'ernie4.5-turbo': partial(
        api.GPT4V,
        model='ernie-4.5-turbo-vl-32k', 
        temperature=0,
        retry=3, 
        max_tokens=12000, 
    ),
    'ernie4.5-a3b': partial(
        api.GPT4V,
        model='ernie-4.5-vl-28b-a3b', 
        temperature=0,
        retry=3, 
        max_tokens=8000,
    ),
    "360zhinao3-vl":
    partial(
        api.GPT4V,
        model="360zhinao3-vl",
        api_base="https://api.360.cn/v1/chat/completions",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        thinking_budget=18000,
        max_tokens=24000,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        retry=3,
    ),
}

api_models['gpt-5'] = cp.deepcopy(api_models['gpt-5-2025-08-07'])
api_models['gpt-5-mini'] = cp.deepcopy(api_models['gpt-5-mini-2025-08-07'])
api_models['gpt-5-nano'] = cp.deepcopy(api_models['gpt-5-nano-2025-08-07'])

emu_series = {
    "emu2_chat": partial(vlm.Emu, model_path="BAAI/Emu2-Chat"),
    "emu3_chat": partial(vlm.Emu3_chat, model_path="BAAI/Emu3-Chat"),
    "emu3_gen": partial(vlm.Emu3_gen, model_path="BAAI/Emu3-Gen"),
}

granite_vision_series = {
    'granite_vision_3.1_2b_preview': partial(vlm.GraniteVision3, model_path="ibm-granite/granite-vision-3.1-2b-preview"),
    'granite_vision_3.2_2b': partial(vlm.GraniteVision3, model_path="ibm-granite/granite-vision-3.2-2b"),
    'granite_vision_3.3_2b': partial(vlm.GraniteVision3, model_path="ibm-granite/granite-vision-3.3-2b"),
}

mmalaya_series = {
    "MMAlaya": partial(vlm.MMAlaya, model_path="DataCanvas/MMAlaya"),
    "MMAlaya2": partial(vlm.MMAlaya2, model_path="DataCanvas/MMAlaya2"),
}

minicpm_series = {
    "MiniCPM-V": partial(vlm.MiniCPM_V, model_path="openbmb/MiniCPM-V"),
    "MiniCPM-V-2": partial(vlm.MiniCPM_V, model_path="openbmb/MiniCPM-V-2"),
    "MiniCPM-Llama3-V-2_5": partial(
        vlm.MiniCPM_Llama3_V, model_path="openbmb/MiniCPM-Llama3-V-2_5"
    ),
    "MiniCPM-V-2_6": partial(vlm.MiniCPM_V_2_6, model_path="openbmb/MiniCPM-V-2_6"),
    "MiniCPM-o-2_6": partial(vlm.MiniCPM_o_2_6, model_path="openbmb/MiniCPM-o-2_6"),
    "MiniCPM-V-4": partial(vlm.MiniCPM_V_4, model_path="openbmb/MiniCPM-V-4"),
    "MiniCPM-V-4_5": partial(vlm.MiniCPM_V_4_5, model_path="openbmb/MiniCPM-V-4_5"),
    "MiniCPM-o-4_5": partial(vlm.MiniCPM_o_4_5, model_path="openbmb/MiniCPM-o-4_5"),
    "MiniCPM-o-4_5_api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='MiniCPM-o-4_5',
        temperature=0.7,
        top_p=0.8,
        top_k=100,
        repetition_penalty=1.02,
        max_tokens=32768,
        retry=6,
        timeout=1800,
    ),
}

xtuner_series = {
    "llava-internlm2-7b": partial(
        vlm.LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-7b",
        llava_path="xtuner/llava-internlm2-7b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm2-20b": partial(
        vlm.LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-20b",
        llava_path="xtuner/llava-internlm2-20b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm-7b": partial(
        vlm.LLaVA_XTuner,
        llm_path="internlm/internlm-chat-7b",
        llava_path="xtuner/llava-internlm-7b",
        visual_select_layer=-2,
        prompt_template="internlm_chat",
    ),
    "llava-v1.5-7b-xtuner": partial(
        vlm.LLaVA_XTuner,
        llm_path="lmsys/vicuna-7b-v1.5",
        llava_path="xtuner/llava-v1.5-7b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-v1.5-13b-xtuner": partial(
        vlm.LLaVA_XTuner,
        llm_path="lmsys/vicuna-13b-v1.5",
        llava_path="xtuner/llava-v1.5-13b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-llama-3-8b": partial(
        vlm.LLaVA_XTuner,
        llm_path="xtuner/llava-llama-3-8b-v1_1",
        llava_path="xtuner/llava-llama-3-8b-v1_1",
        visual_select_layer=-2,
        prompt_template="llama3_chat",
    ),
}

qwen_series = {
    "qwen_base": partial(vlm.QwenVL, model_path="Qwen/Qwen-VL"),
    "qwen_chat": partial(vlm.QwenVLChat, model_path="Qwen/Qwen-VL-Chat"),
    "monkey": partial(vlm.Monkey, model_path="echo840/Monkey"),
    "monkey-chat": partial(vlm.MonkeyChat, model_path="echo840/Monkey-Chat"),
    "minimonkey": partial(vlm.MiniMonkey, model_path="mx262/MiniMonkey"),
}

thyme_series = {
    "Thyme-7B": partial(vlm.Thyme, model_path="Kwai-Keye/Thyme-RL")
}

llava_series = {
    "llava_v1.5_7b": partial(vlm.LLaVA, model_path="liuhaotian/llava-v1.5-7b"),
    "llava_v1.5_13b": partial(vlm.LLaVA, model_path="liuhaotian/llava-v1.5-13b"),
    "llava_v1_7b": partial(vlm.LLaVA, model_path=LLAVA_V1_7B_MODEL_PTH),
    "sharegpt4v_7b": partial(vlm.LLaVA, model_path="Lin-Chen/ShareGPT4V-7B"),
    "sharegpt4v_13b": partial(vlm.LLaVA, model_path="Lin-Chen/ShareGPT4V-13B"),
    "llava_next_vicuna_7b": partial(
        vlm.LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-7b-hf"
    ),
    "llava_next_vicuna_13b": partial(
        vlm.LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-13b-hf"
    ),
    "llava_next_mistral_7b": partial(
        vlm.LLaVA_Next, model_path="llava-hf/llava-v1.6-mistral-7b-hf"
    ),
    "llava_next_yi_34b": partial(vlm.LLaVA_Next, model_path="llava-hf/llava-v1.6-34b-hf"),
    "llava_next_llama3": partial(
        vlm.LLaVA_Next, model_path="llava-hf/llama3-llava-next-8b-hf"
    ),
    "llava_next_72b": partial(vlm.LLaVA_Next, model_path="llava-hf/llava-next-72b-hf"),
    "llava_next_110b": partial(vlm.LLaVA_Next, model_path="llava-hf/llava-next-110b-hf"),
    "llava_next_qwen_32b": partial(
        vlm.LLaVA_Next2, model_path="lmms-lab/llava-next-qwen-32b"
    ),
    "llava_next_interleave_7b": partial(
        vlm.LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-hf"
    ),
    "llava_next_interleave_7b_dpo": partial(
        vlm.LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-dpo-hf"
    ),
    "llava-onevision-qwen2-0.5b-ov-hf": partial(
        vlm.LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    ),
    "llava-onevision-qwen2-0.5b-si-hf": partial(
        vlm.LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    ),
    "llava-onevision-qwen2-7b-ov-hf": partial(
        vlm.LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    ),
    "llava-onevision-qwen2-7b-si-hf": partial(
        vlm.LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-si-hf"
    ),
    "llava_onevision_qwen2_0.5b_si": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-si"
    ),
    "llava_onevision_qwen2_7b_si": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-si"
    ),
    "llava_onevision_qwen2_72b_si": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-si"
    ),
    "llava_onevision_qwen2_0.5b_ov": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-ov"
    ),
    "llava_onevision_qwen2_7b_ov": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-ov"
    ),
    "llava_onevision_qwen2_72b_ov": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-ov-sft"
    ),
    "Aquila-VL-2B": partial(vlm.LLaVA_OneVision, model_path="BAAI/Aquila-VL-2B-llava-qwen"),
    "llava_video_qwen2_7b": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
    ),
    "llava_video_qwen2_72b": partial(
        vlm.LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-72B-Qwen2"
    ),
    "LLaVA-OneVision-1.5-8B-Instruct": partial(
        vlm.LLaVA_OneVision_1_5, model_path="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct", max_new_tokens=8192
    ),
}

varco_vision_series = {
    "varco-vision-hf": partial(
        vlm.LLaVA_OneVision_HF, model_path="NCSOFT/VARCO-VISION-14B-HF"
    ),
    "varco-vision-2-1.7b": partial(
        vlm.VarcoVision, model_path="NCSOFT/VARCO-VISION-2.0-1.7B"
    ),
    "varco-vision-2-14b": partial(
        vlm.VarcoVision, model_path="NCSOFT/VARCO-VISION-2.0-14B"
    ),
}

vita_series = {
    "vita": partial(vlm.VITA, model_path="VITA-MLLM/VITA", root=VITA_ROOT),
    "vita_qwen2": partial(vlm.VITAQwen2, model_path="VITA-MLLM/VITA-1.5", root=VITA_ROOT),
}

long_vita_series = {
    "Long-VITA-16K": partial(
        vlm.LongVITA, model_path="VITA-MLLM/Long-VITA-16K_HF", max_num_frame=128
    ),
    "Long-VITA-128K": partial(
        vlm.LongVITA, model_path="VITA-MLLM/Long-VITA-128K_HF", max_num_frame=256
    ),
    "Long-VITA-1M": partial(
        vlm.LongVITA, model_path="VITA-MLLM/Long-VITA-1M_HF", max_num_frame=256
    ),
}

interns1_mini = {
    "Intern-S1-mini": partial(
        vlm.InternS1Chat, model_path="/mnt/shared-storage-user/mllm/lijinsong/models/Intern-S1-mini/"
    ),
}

internvl = {
    "InternVL-Chat-V1-1": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-1", version="V1.1"
    ),
    "InternVL-Chat-V1-2": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2", version="V1.2"
    ),
    "InternVL-Chat-V1-2-Plus": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2-Plus", version="V1.2"
    ),
    "InternVL-Chat-V1-5": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL-Chat-V1-5",
        version="V1.5",
    )
}

mini_internvl = {
    "Mini-InternVL-Chat-2B-V1-5": partial(
        vlm.InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-2B-V1-5", version="V1.5"
    ),
    "Mini-InternVL-Chat-4B-V1-5": partial(
        vlm.InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-4B-V1-5", version="V1.5"
    ),
}

internvl2 = {
    "InternVL2-1B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-1B", version="V2.0"
    ),
    "InternVL2-2B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-2B", version="V2.0"
    ),
    "InternVL2-4B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-4B", version="V2.0"
    ),
    "InternVL2-8B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-8B", version="V2.0"
    ),
    "InternVL2-26B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-26B", version="V2.0"
    ),
    "InternVL2-40B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-40B", version="V2.0"
    ),
    "InternVL2-76B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-Llama3-76B", version="V2.0"
    ),
    "InternVL2-8B-MPO": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2-8B-MPO", version="V2.0"
    ),
    "InternVL2-8B-MPO-CoT": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
}

internvl2_5 = {
    "InternVL2_5-1B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-1B", version="V2.0"
    ),
    "InternVL2_5-2B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-2B", version="V2.0"
    ),
    "QTuneVL1-2B": partial(
        vlm.InternVLChat, model_path="hanchaow/QTuneVL1-2B", version="V2.0"
    ),
    "InternVL2_5-4B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-4B", version="V2.0"
    ),
    "InternVL2_5-8B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0"
    ),
    "InternVL2_5-26B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-26B", version="V2.0"
    ),
    "InternVL2_5-38B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-38B", version="V2.0"
    ),
    "InternVL2_5-78B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-78B", version="V2.0"
    ),
    # InternVL2.5 series with Best-of-N evaluation
    "InternVL2_5-8B-BoN-8": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0",
        best_of_n=8, reward_model_path="OpenGVLab/VisualPRM-8B",
    ),
}

internvl2_5_mpo = {
    "InternVL2_5-1B-MPO": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2_5-1B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-2B-MPO": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2_5-2B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-4B-MPO": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2_5-4B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-MPO": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2_5-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-26B-MPO": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2_5-26B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-38B-MPO": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2_5-38B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-78B-MPO": partial(
        vlm.InternVLChat,
        model_path="OpenGVLab/InternVL2_5-78B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-GUI": partial(
        vlm.InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/models/internvl2_5_8b_internlm2_5_7b_dynamic_res_stage1", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
     "InternVL3-7B-GUI": partial(
        vlm.InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/GUI/checkpoints/internvl3_7b_dynamic_res_stage1_56/", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
}

internvl3 = {
    "InternVL3-1B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3-1B", version="V2.0"
    ),
    "InternVL3-2B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3-2B", version="V2.0"
    ),
    "InternVL3-8B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3-8B", version="V2.0",
    ),
    "InternVL3-9B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3-9B", version="V2.0"
    ),
    "InternVL3-14B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3-14B", version="V2.0"
    ),
    "InternVL3-38B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3-38B", version="V2.0"
    ),
    "InternVL3-78B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3-78B", version="V2.0"
    ),
}

internvl3_5 = {
    "InternVL3_5-1B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-1B", version="V2.0"
    ),
    "InternVL3_5-2B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-2B", version="V2.0"
    ),
    "InternVL3_5-4B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-4B", version="V2.0"
    ),
    "InternVL3_5-8B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-8B", version="V2.0"
    ),
    "InternVL3_5-14B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-14B", version="V2.0"
    ),
    "InternVL3_5-GPT-OSS-20B-A4B-Preview": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview", version="V2.0"
    ),
    "InternVL3_5-30B-A3B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-30B-A3B", version="V2.0"
    ),
    "InternVL3_5-38B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-38B", version="V2.0"
    ),
    "InternVL3_5-241B-A28B": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-241B-A28B", version="V2.0",
        max_new_tokens=16384,
    ),

    "InternVL3_5-1B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-1B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-2B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-2B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-4B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-4B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-8B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-8B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-14B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-14B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-GPT-OSS-20B-A4B-Preview-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-30B-A3B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-30B-A3B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-38B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-38B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-241B-A28B-Thinking": partial(
        vlm.InternVLChat, model_path="OpenGVLab/InternVL3_5-241B-A28B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-241B-A28B-Thinking-api": partial(
        api.LMDeployAPI, model="internvl-3.5-241b", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
}

qwen3vl_series = {
    "Qwen3-VL-235B-A22B-Instruct": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-235B-A22B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-235B-A22B-Thinking": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-235B-A22B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-30B-A3B-Instruct": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-30B-A3B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-30B-A3B-Thinking": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-30B-A3B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-8B-Thinking": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-8B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-4B-Thinking": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-4B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-8B-Instruct": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B-Instruct": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-4B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-2B-Instruct": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-2B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-32B-Instruct": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-32B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20

    ),
    "Qwen3-VL-2B-Thinking": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-2B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-32B-Thinking": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-32B-Thinking",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-Omni-30B-A3B-Instruct": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    "Qwen3-Omni-30B-A3B-Thinking": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    "Qwen3-Omni-30B-A3B-Captioner": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    
}

qwen3_5_series = {
    # vllm serve command example: 
    # vllm serve Qwen/Qwen3.5-122B-A10B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3
    "Qwen3.5-35B-A3B_api": partial(
        api.LMDeployAPI,
        model="Qwen/Qwen3.5-122B-A10B",
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_new_tokens=32768,
        retry=6,
        timeout=1800,
    ),
    "Qwen3.5-397B-A17B_api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='Qwen3.5-397B',
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=32768,
        retry=10,
        timeout=1800,
    ),
    "Qwen3.5-122B-A10B_ThinkMode_api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='Qwen3.5-35B',
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=81920,
        retry=10,
        timeout=900,
    ),
    "Qwen3.5-122B-A10B_InstructMode_api": partial(
        api.LMDeployAPI,
        model="Qwen/Qwen3.5-122B-A10B",
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=81920,
        retry=10,
        timeout=900,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "Qwen3.5-397B-A17B": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3.5-397B-A17B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-122B-A10B": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3.5-122B-A10B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-35B-A3B": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3.5-35B-A3B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-27B": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3.5-27B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
}

sail_series = {
    "SAIL-VL-2B": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL-2B"),
    "SAIL-VL-1.5-2B": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d5-2B", use_msac = True),
    "SAIL-VL-1.5-8B": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d5-8B", use_msac = True),
    "SAIL-VL-1.6-8B": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d6-8B", use_msac = True),
    "SAIL-VL-1.7-Thinking-2B-2507": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d7-Thinking-2B-2507", use_msac = True, use_cot=True, max_new_tokens=4096),
    "SAIL-VL-1.7-Thinking-8B-2507": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d7-Thinking-8B-2507", use_msac = True, use_cot=True, max_new_tokens=4096),
    "SAIL-VL2-2B": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL2-2B", use_msac = True),
    "SAIL-VL2-8B": partial(vlm.SailVL, model_path="BytedanceDouyinContent/SAIL-VL2-8B", use_msac = True),
}

ristretto_series = {
    "Ristretto-3B": partial(vlm.Ristretto, model_path="LiAutoAD/Ristretto-3B"),
}

yivl_series = {
    "Yi_VL_6B": partial(vlm.Yi_VL, model_path="01-ai/Yi-VL-6B", root=Yi_ROOT),
    "Yi_VL_34B": partial(vlm.Yi_VL, model_path="01-ai/Yi-VL-34B", root=Yi_ROOT),
}

xcomposer_series = {
    "XComposer": partial(vlm.XComposer, model_path="internlm/internlm-xcomposer-vl-7b"),
    "sharecaptioner": partial(vlm.ShareCaptioner, model_path="Lin-Chen/ShareCaptioner"),
    "XComposer2": partial(vlm.XComposer2, model_path="internlm/internlm-xcomposer2-vl-7b"),
    "XComposer2_1.8b": partial(
        vlm.XComposer2, model_path="internlm/internlm-xcomposer2-vl-1_8b"
    ),
    "XComposer2_4KHD": partial(
        vlm.XComposer2_4KHD, model_path="internlm/internlm-xcomposer2-4khd-7b"
    ),
    "XComposer2d5": partial(
        vlm.XComposer2d5, model_path="internlm/internlm-xcomposer2d5-7b"
    ),
}

minigpt4_series = {
    "MiniGPT-4-v2": partial(vlm.MiniGPT4, mode="v2", root=MiniGPT4_ROOT),
    "MiniGPT-4-v1-7B": partial(vlm.MiniGPT4, mode="v1_7b", root=MiniGPT4_ROOT),
    "MiniGPT-4-v1-13B": partial(vlm.MiniGPT4, mode="v1_13b", root=MiniGPT4_ROOT),
}

idefics_series = {
    "idefics_9b_instruct": partial(
        vlm.IDEFICS, model_path="HuggingFaceM4/idefics-9b-instruct"
    ),
    "idefics_80b_instruct": partial(
        vlm.IDEFICS, model_path="HuggingFaceM4/idefics-80b-instruct"
    ),
    "idefics2_8b": partial(vlm.IDEFICS2, model_path="HuggingFaceM4/idefics2-8b"),
    # Idefics3 follows Idefics2 Pattern
    "Idefics3-8B-Llama3": partial(
        vlm.IDEFICS2, model_path="HuggingFaceM4/Idefics3-8B-Llama3"
    ),
    'granite-docling-258M': partial(
        vlm.DOCLING, model_path="ibm-granite/granite-docling-258M"
    )

}

smolvlm_series = {
    "SmolVLM-256M": partial(vlm.SmolVLM, model_path="HuggingFaceTB/SmolVLM-256M-Instruct"),
    "SmolVLM-500M": partial(vlm.SmolVLM, model_path="HuggingFaceTB/SmolVLM-500M-Instruct"),
    "SmolVLM": partial(vlm.SmolVLM, model_path="HuggingFaceTB/SmolVLM-Instruct"),
    "SmolVLM-DPO": partial(vlm.SmolVLM, model_path="HuggingFaceTB/SmolVLM-Instruct-DPO"),
    "SmolVLM-Synthetic": partial(vlm.SmolVLM, model_path="HuggingFaceTB/SmolVLM-Synthetic"),
    "SmolVLM2-256M": partial(
        vlm.SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    ),
    "SmolVLM2-500M": partial(
        vlm.SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    ),
    "SmolVLM2": partial(vlm.SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
}

instructblip_series = {
    "instructblip_7b": partial(vlm.InstructBLIP, name="instructblip_7b"),
    "instructblip_13b": partial(vlm.InstructBLIP, name="instructblip_13b"),
}

deepseekvl_series = {
    "deepseek_vl_7b": partial(vlm.DeepSeekVL, model_path="deepseek-ai/deepseek-vl-7b-chat"),
    "deepseek_vl_1.3b": partial(
        vlm.DeepSeekVL, model_path="deepseek-ai/deepseek-vl-1.3b-chat"
    ),
}

deepseekvl2_series = {
    "deepseek_vl2_tiny": partial(
        vlm.DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-tiny"
    ),
    "deepseek_vl2_small": partial(
        vlm.DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-small"
    ),
    "deepseek_vl2": partial(vlm.DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2"),
}

deepseekocr_series = {
    "DeepSeek-OCR": partial(
        vlm.DeepSeekOCR, model_path="deepseek-ai/DeepSeek-OCR"
    ),
}

janus_series = {
    "Janus-1.3B": partial(vlm.Janus, model_path="deepseek-ai/Janus-1.3B"),
    "Janus-Pro-1B": partial(vlm.Janus, model_path="deepseek-ai/Janus-Pro-1B"),
    "Janus-Pro-7B": partial(vlm.Janus, model_path="deepseek-ai/Janus-Pro-7B"),
}

cogvlm_series = {
    "cogvlm-grounding-generalist": partial(
        vlm.CogVlm,
        model_path="THUDM/cogvlm-grounding-generalist-hf",
        tokenizer_name="lmsys/vicuna-7b-v1.5",
    ),
    "cogvlm-chat": partial(
        vlm.CogVlm, model_path="THUDM/cogvlm-chat-hf", tokenizer_name="lmsys/vicuna-7b-v1.5"
    ),
    "cogvlm2-llama3-chat-19B": partial(
        vlm.CogVlm, model_path="THUDM/cogvlm2-llama3-chat-19B"
    ),
    "glm-4v-9b": partial(vlm.GLM4v, model_path="THUDM/glm-4v-9b"),
    "GLM4_1VThinking-9b": partial(vlm.GLMThinking, model_path="THUDM/GLM-4.1V-9B-Thinking"),
    "GLM4_5V": partial(vlm.GLMThinking, model_path="THUDM/GLM-4.5V"),
    "GLM4_6V": partial(vlm.GLMThinking, model_path="THUDM/GLM-4.6V"),
    "GLM4_6V-api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='glm-4.6v',
        temperature=0.8,
        top_p=0.6,
        top_k=2,
        repetition_penalty=1.1,
        max_tokens=16384,
        retry=6,
        timeout=1800,
    ),
}

wemm_series = {
    "WeMM": partial(vlm.WeMM, model_path="feipengma/WeMM"),
}

cambrian_series = {
    "cambrian_8b": partial(vlm.Cambrian, model_path="nyu-visionx/cambrian-8b"),
    "cambrian_13b": partial(vlm.Cambrian, model_path="nyu-visionx/cambrian-13b"),
    "cambrian_34b": partial(vlm.Cambrian, model_path="nyu-visionx/cambrian-34b"),
    
    "cambrian-s-0.5b": partial(vlm.CambrianS, model_path="nyu-visionx/Cambrian-S-0.5B"),
    "cambrian-s-1.5b": partial(vlm.CambrianS, model_path="nyu-visionx/Cambrian-S-1.5B"),
    "cambrian-s-3b": partial(vlm.CambrianS, model_path="nyu-visionx/Cambrian-S-3B"),
    "cambrian-s-7b": partial(vlm.CambrianS, model_path="nyu-visionx/Cambrian-S-7B"),
}

chameleon_series = {
    "chameleon_7b": partial(vlm.Chameleon, model_path="facebook/chameleon-7b"),
    "chameleon_30b": partial(vlm.Chameleon, model_path="facebook/chameleon-30b"),
}

vila_series = {
    "VILA1.5-3b": partial(vlm.VILA, model_path="Efficient-Large-Model/VILA1.5-3b"),
    "Llama-3-VILA1.5-8b": partial(
        vlm.VILA, model_path="Efficient-Large-Model/Llama-3-VILA1.5-8b"
    ),
    "VILA1.5-13b": partial(vlm.VILA, model_path="Efficient-Large-Model/VILA1.5-13b"),
    "VILA1.5-40b": partial(vlm.VILA, model_path="Efficient-Large-Model/VILA1.5-40b"),
    "NVILA-8B": partial(vlm.NVILA, model_path="Efficient-Large-Model/NVILA-8B"),
    "NVILA-15B": partial(vlm.NVILA, model_path="Efficient-Large-Model/NVILA-15B"),
}

ovis_series = {
    "Ovis1.5-Llama3-8B": partial(vlm.Ovis, model_path="AIDC-AI/Ovis1.5-Llama3-8B"),
    "Ovis1.5-Gemma2-9B": partial(vlm.Ovis, model_path="AIDC-AI/Ovis1.5-Gemma2-9B"),
    "Ovis1.6-Gemma2-9B": partial(vlm.Ovis1_6, model_path="AIDC-AI/Ovis1.6-Gemma2-9B"),
    "Ovis1.6-Llama3.2-3B": partial(vlm.Ovis1_6, model_path="AIDC-AI/Ovis1.6-Llama3.2-3B"),
    "Ovis1.6-Gemma2-27B": partial(
        vlm.Ovis1_6_Plus, model_path="AIDC-AI/Ovis1.6-Gemma2-27B"
    ),
    "Ovis2-1B": partial(vlm.Ovis2, model_path="AIDC-AI/Ovis2-1B"),
    "Ovis2-2B": partial(vlm.Ovis2, model_path="AIDC-AI/Ovis2-2B"),
    "Ovis2-4B": partial(vlm.Ovis2, model_path="AIDC-AI/Ovis2-4B"),
    "Ovis2-8B": partial(vlm.Ovis2, model_path="AIDC-AI/Ovis2-8B"),
    "Ovis2-16B": partial(vlm.Ovis2, model_path="AIDC-AI/Ovis2-16B"),
    "Ovis2-34B": partial(vlm.Ovis2, model_path="AIDC-AI/Ovis2-34B"),
    "Ovis-U1-3B": partial(vlm.OvisU1, model_path="AIDC-AI/Ovis-U1-3B"),
    "Ovis2.5-2B": partial(vlm.Ovis2_5, model_path="AIDC-AI/Ovis2.5-2B"),
    "Ovis2.5-9B": partial(vlm.Ovis2_5, model_path="AIDC-AI/Ovis2.5-9B"),
    "Ovis2.6-30B-A3B_api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='Ovis2.6-30B-A3B',
        max_tokens=32768,
        retry=6,
        timeout=1800,
    ),
}

mantis_series = {
    "Mantis-8B-siglip-llama3": partial(
        vlm.Mantis, model_path="TIGER-Lab/Mantis-8B-siglip-llama3"
    ),
    "Mantis-8B-clip-llama3": partial(
        vlm.Mantis, model_path="TIGER-Lab/Mantis-8B-clip-llama3"
    ),
    "Mantis-8B-Idefics2": partial(vlm.Mantis, model_path="TIGER-Lab/Mantis-8B-Idefics2"),
    "Mantis-8B-Fuyu": partial(vlm.Mantis, model_path="TIGER-Lab/Mantis-8B-Fuyu"),
}

phi3_series = {
    "Phi-3-Vision": partial(
        vlm.Phi3Vision, model_path="microsoft/Phi-3-vision-128k-instruct"
    ),
    "Phi-3.5-Vision": partial(
        vlm.Phi3_5Vision, model_path="microsoft/Phi-3.5-vision-instruct"
    ),
}

phi4_series = {
    'Phi-4-Vision': partial(vlm.Phi4Multimodal, model_path='microsoft/Phi-4-multimodal-instruct'),
}

xgen_mm_series = {
    "xgen-mm-phi3-interleave-r-v1.5": partial(
        vlm.XGenMM, model_path="Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    ),
    "xgen-mm-phi3-dpo-r-v1.5": partial(
        vlm.XGenMM, model_path="Salesforce/xgen-mm-phi3-mini-instruct-dpo-r-v1.5"
    ),
}

hawkvl_series = {
    "HawkVL-2B": partial(
        vlm.HawkVL,
        model_path="xjtupanda/HawkVL-2B",
        min_pixels=4 * 28 * 28,
        max_pixels=6800 * 28 * 28,
        use_custom_prompt=True
    )
}

qwen2vl_series = {
    "Qwen-VL-Max-20250813": partial(
        api.Qwen2VLAPI,
        model="qwen-vl-max-2025-08-13",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_length=8192,
    ),
    "Qwen-VL-Max-0809": partial(
        api.Qwen2VLAPI,
        model="qwen-vl-max-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen-VL-Plus-0809": partial(
        api.Qwen2VLAPI,
        model="qwen-vl-plus-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "QVQ-72B-Preview": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/QVQ-72B-Preview",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        system_prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
        max_new_tokens=8192,
        post_process=False,
    ),
    "Qwen2-VL-72B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-AWQ": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int4": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int8": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-AWQ": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int4": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int8": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "XinYuan-VL-2B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Cylingo/Xinyuan-VL-2B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2.5-VL-3B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-3B-Instruct-AWQ": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-ForVideo": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-AWQ": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-32B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-32B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MiMo-VL-7B-SFT": partial(
        vlm.Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "MiMo-VL-7B-RL": partial(
        vlm.Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-RL",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "Qwen2.5-VL-72B-Instruct-ForVideo": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct-AWQ": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-Omni-7B-ForVideo": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
        use_audio_in_video=True, # set use audio in video
    ),
    "Qwen2.5-Omni-7B": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    # Qwen3-VL series (local inference)
    "Qwen3-VL-8B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_vllm=True,
    ),
    "Qwen3-VL-30B-A3B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen3-VL-30B-A3B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_vllm=True,
    ),
    "Qwen3-VL-235B-A22B-Instruct": partial(
        vlm.Qwen2VLChat,
        model_path="Qwen/Qwen3-VL-235B-A22B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_vllm=True,
    ),
    'VLM-R1': partial(
        vlm.VLMR1Chat, 
        model_path='omlab/VLM-R1-Qwen2.5VL-3B-Math-0305', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False),
    'VLAA-Thinker-Qwen2.5VL-3B': partial(
        vlm.VLAAThinkerChat, 
        model_path='UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False,
        post_process=True, # post processing for evaluation
        system_prompt=(''
                    "You are VL-Thinking🤔, a helpful assistant with excellent reasoning ability."
                    " A user asks you a question, and you should try to solve it."
                    " You should first think about the reasoning process in the mind and then provides the user with the answer."
                    " The reasoning process and answer are enclosed within <think> </think> and"
                    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
                    "<answer> answer here </answer>"
                ),
    ),
    'VLAA-Thinker-Qwen2.5VL-7B': partial(
        vlm.VLAAThinkerChat, 
        model_path='UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False,
        post_process=True, # post processing for evaluation
        system_prompt=(''
                    "You are VL-Thinking🤔, a helpful assistant with excellent reasoning ability."
                    " A user asks you a question, and you should try to solve it."
                    " You should first think about the reasoning process in the mind and then provides the user with the answer."
                    " The reasoning process and answer are enclosed within <think> </think> and"
                    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
                    "<answer> answer here </answer>"
                ),
    ),
    'WeThink-Qwen2.5VL-7B': partial(
        vlm.WeThinkVL, 
        model_path='yangjie-cv/WeThink-Qwen2.5VL-7B', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False,
        system_prompt=("You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."
        ),
    ),
}

slime_series = {
    "Slime-7B": partial(vlm.SliME, model_path="yifanzhang114/SliME-vicuna-7B"),
    "Slime-8B": partial(vlm.SliME, model_path="yifanzhang114/SliME-Llama3-8B"),
    "Slime-13B": partial(vlm.SliME, model_path="yifanzhang114/SliME-vicuna-13B"),
}

eagle_series = {
    "Eagle-X4-8B-Plus": partial(vlm.Eagle, model_path="NVEagle/Eagle-X4-8B-Plus"),
    "Eagle-X4-13B-Plus": partial(vlm.Eagle, model_path="NVEagle/Eagle-X4-13B-Plus"),
    "Eagle-X5-7B": partial(vlm.Eagle, model_path="NVEagle/Eagle-X5-7B"),
    "Eagle-X5-13B": partial(vlm.Eagle, model_path="NVEagle/Eagle-X5-13B"),
    "Eagle-X5-13B-Chat": partial(vlm.Eagle, model_path="NVEagle/Eagle-X5-13B-Chat"),
    "Eagle-X5-34B-Chat": partial(vlm.Eagle, model_path="NVEagle/Eagle-X5-34B-Chat"),
    "Eagle-X5-34B-Plus": partial(vlm.Eagle, model_path="NVEagle/Eagle-X5-34B-Plus"),
}

moondream_series = {
    "Moondream1": partial(vlm.Moondream1, model_path="vikhyatk/moondream1"),
    "Moondream2": partial(vlm.Moondream2, model_path="vikhyatk/moondream2"),
    "Moondream3": partial(vlm.Moondream3, model_path="moondream/moondream3-preview"),
}

llama_series = {
    "Llama-3.2-11B-Vision-Instruct": partial(
        vlm.llama_vision, model_path="meta-llama/Llama-3.2-11B-Vision-Instruct"
    ),
    "LLaVA-CoT": partial(vlm.llama_vision, model_path="Xkev/Llama-3.2V-11B-cot"),
    "Llama-3.2-90B-Vision-Instruct": partial(
        vlm.llama_vision, model_path="meta-llama/Llama-3.2-90B-Vision-Instruct"
    ),
    "Llama-4-Scout-17B-16E-Instruct": partial(
        vlm.llama4, model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct", use_vllm=True
    ),
}

molmo_series = {
    "molmoE-1B-0924": partial(vlm.molmo, model_path="allenai/MolmoE-1B-0924"),
    "molmo-7B-D-0924": partial(vlm.molmo, model_path="allenai/Molmo-7B-D-0924"),
    "molmo-7B-O-0924": partial(vlm.molmo, model_path="allenai/Molmo-7B-O-0924"),
    "molmo-72B-0924": partial(vlm.molmo, model_path="allenai/Molmo-72B-0924"),
}

kosmos_series = {
    "Kosmos2": partial(vlm.Kosmos2, model_path="microsoft/kosmos-2-patch14-224")
}

points_series = {
    "POINTS-Yi-1.5-9B-Chat": partial(
        vlm.POINTS, model_path="WePOINTS/POINTS-Yi-1-5-9B-Chat"
    ),
    "POINTS-Qwen-2.5-7B-Chat": partial(
        vlm.POINTS, model_path="WePOINTS/POINTS-Qwen-2-5-7B-Chat"
    ),
    "POINTSV15-Qwen-2.5-7B-Chat": partial(
        vlm.POINTSV15, model_path="WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat"
    ),
}

nvlm_series = {
    "NVLM": partial(vlm.NVLM, model_path="nvidia/NVLM-D-72B"),
    "NVLM-D-72B_api": partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='NVLM-D-72B',
        top_p=1.0,
        max_new_tokens=32768,
        retry=6,
        timeout=1800,
    ),
}

vintern_series = {
    "Vintern-3B-beta": partial(vlm.VinternChat, model_path="5CD-AI/Vintern-3B-beta"),
    "Vintern-1B-v2": partial(vlm.VinternChat, model_path="5CD-AI/Vintern-1B-v2"),
}

aria_series = {"Aria": partial(vlm.Aria, model_path="rhymes-ai/Aria")}

h2ovl_series = {
    "h2ovl-mississippi-2b": partial(vlm.H2OVLChat, model_path="h2oai/h2ovl-mississippi-2b"),
    "h2ovl-mississippi-1b": partial(
        vlm.H2OVLChat, model_path="h2oai/h2ovl-mississippi-800m"
    ),
}

valley_series = {
    "valley2": partial(
        vlm.Valley2Chat, model_path="bytedance-research/Valley-Eagle-7B"
    ),
    "valley2_dpo": partial(
        vlm.Valley2Chat, model_path="bytedance-research/Valley2-DPO"
    ),
    "valley2.5": partial(
        vlm.Valley3Chat, use_gthinker_thinking=True, model_path="bytedance-research/Valley2.5"
    ),
}

ola_series = {
    "ola": partial(vlm.Ola, model_path="THUdyh/Ola-7b"),
}

xvl_series = {
    "X-VL-4B": partial(vlm.X_VL_HF, model_path="YannQi/X-VL-4B", temperature=0, retry=10),
}

ross_series = {
    "ross-qwen2-7b": partial(vlm.Ross, model_path="HaochenWang/ross-qwen2-7b"),
}

ursa_series = {
    "URSA-8B": partial(vlm.UrsaChat, model_path="URSA-MATH/URSA-8B"),
    "URSA-8B-PS-GRPO": partial(vlm.UrsaChat, model_path="URSA-MATH/URSA-8B-PS-GRPO")    
}

gemma_series = {
    "paligemma-3b-mix-448": partial(
        vlm.PaliGemma, model_path="google/paligemma-3b-mix-448"
    ),
    
    # 3B
    "paligemma2-3b-pt-224":  partial(vlm.PaliGemma, model_path="google/paligemma2-3b-pt-224"),
    "paligemma2-3b-pt-448":  partial(vlm.PaliGemma, model_path="google/paligemma2-3b-pt-448"),
    "paligemma2-3b-mix-224": partial(vlm.PaliGemma, model_path="google/paligemma2-3b-mix-224"),
    "paligemma2-3b-mix-448": partial(vlm.PaliGemma, model_path="google/paligemma2-3b-mix-448"),

    # 10B
    "paligemma2-10b-pt-224":  partial(vlm.PaliGemma, model_path="google/paligemma2-10b-pt-224"),
    "paligemma2-10b-pt-448":  partial(vlm.PaliGemma, model_path="google/paligemma2-10b-pt-448"),
    "paligemma2-10b-mix-224": partial(vlm.PaliGemma, model_path="google/paligemma2-10b-mix-224"),
    "paligemma2-10b-mix-448": partial(vlm.PaliGemma, model_path="google/paligemma2-10b-mix-448"),

    # 28B
    "paligemma2-28b-pt-224":  partial(vlm.PaliGemma, model_path="google/paligemma2-28b-pt-224"),
    "paligemma2-28b-pt-448":  partial(vlm.PaliGemma, model_path="google/paligemma2-28b-pt-448"),
    "paligemma2-28b-mix-224": partial(vlm.PaliGemma, model_path="google/paligemma2-28b-mix-224"),
    "paligemma2-28b-mix-448": partial(vlm.PaliGemma, model_path="google/paligemma2-28b-mix-448"),

    'Gemma3-4B': partial(vlm.Gemma3, model_path='google/gemma-3-4b-it'),
    'Gemma3-12B': partial(vlm.Gemma3, model_path='google/gemma-3-12b-it'),
    'Gemma3-27B': partial(vlm.Gemma3, model_path='google/gemma-3-27b-it')
}

aguvis_series = {
    "aguvis_7b": partial(
        vlm.Qwen2VLChatAguvis,
        model_path=os.getenv(
            "EVAL_MODEL",
            "xlangai/Aguvis-7B-720P",
        ),
        min_pixels=256 * 28 * 28,
        max_pixels=46 * 26 * 28 * 28,
        use_custom_prompt=False,
        mode='grounding',
    )
}

kimi_series = {
    'Kimi-VL-A3B-Thinking': partial(vlm.KimiVL, model_path='moonshotai/Kimi-VL-A3B-Thinking'),
    'Kimi-VL-A3B-Instruct': partial(vlm.KimiVL, model_path='moonshotai/Kimi-VL-A3B-Instruct'),
    'Kimi-VL-A3B-Thinking-2506': partial(vlm.KimiVL, model_path='moonshotai/Kimi-VL-A3B-Thinking-2506', temperature=0.8, max_tokens=32768, extract_summary=True),
    'Kimi-K2.5-api': partial(
        api.LMDeployAPI,
        api_base="http://0.0.0.0:8000/v1/chat/completions",
        model='Kimi-K2.5',
        temperature=1.0,
        top_p=0.95,
        max_tokens=32768,
        retry=6,
        timeout=1800,
    )
}

flash_vl = {
    'Flash-VL-2B-Dynamic-ISS': partial(vlm.FlashVL, model_path='FlashVL/FlashVL-2B-Dynamic-ISS')
}


oryx_series = {
    'oryx': partial(vlm.Oryx, model_path="THUdyh/Oryx-1.5-7B"),
}

# recommend: vllm serve moonshotai/Kimi-VL-A3B-Thinking-2506 
# --served-model-name api-kimi-vl-thinking-2506 --trust-remote-code
# --tensor-parallel-size 2 --max-num-batched-tokens 131072 
# --max-model-len 131072 --limit-mm-per-prompt image=256
kimi_vllm_series = {
    "api-kimi-vl-thinking-2506": partial(
        api.KimiVLAPI,
        model="api-kimi-vl-thinking-2506",
    ),
    "api-kimi-vl-thinking": partial(
        api.KimiVLAPI,
        model="api-kimi-vl-thinking",
    ),
    "api-kimi-vl": partial(
        api.KimiVLAPI,
        model="api-kimi-vl",
        max_new_tokens=2048,
        temperature=0,
    ),
}


treevgr_series = {
    'TreeVGR-7B': partial(
        vlm.TreeVGR, 
        model_path='HaochenWang/TreeVGR-7B',
        min_pixels=1280*28*28, max_pixels=16384*28*28,
    ),
}

# QTuneVL series
qtunevl_series = {
    "QTuneVL1_5-2B": partial(
        vlm.QTuneVLChat, model_path="hanchaow/QTuneVL1_5-2B", version="V1.5"
    ),

    "QTuneVL1_5-3B": partial(
        vlm.QTuneVL,
        model_path="hanchaow/QTuneVL1_5-3B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=True,
        post_process=True
    ),
}

# RbdashMM series via lmdeploy API
rbdashmm_api_series_lmdeploy = {
    "rbdashmm3_DPO_38B_api": partial(
        api.RBdashMMChat3_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    ),
    "rbdashmm3_5_DPO_38B_api": partial(
        api.RBdashChat3_5_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    ),
    "rbdashmm3_5_38B_api": partial(
        api.RBdashMMChat3_5_38B_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    ),
    "rbdashmm3_78B_api": partial(
        api.RBdashMMChat3_78B_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    )
}

logics_series = {
    "Logics-Thinking-8B": partial(vlm.Logics_Thinking,model_path='Logics-MLLM/Logics-Thinking-8B'),
    "Logics-Thinking-32B": partial(vlm.Logics_Thinking,model_path='Logics-MLLM/Logics-Thinking-32B'),
}

insight_v_series = {
    "insightv": partial(vlm.InsightV, pretrained_reason="THUdyh/Insight-V-Reason-LLaMA3", pretrained_summary="THUdyh/Insight-V-Summary-LLaMA3"),
}

cosmos_series = {
    'Cosmos-Reason1-7B': partial(vlm.Cosmos, model_path='nvidia/Cosmos-Reason1-7B', use_vllm=True),
}

keye_series = {
    "Keye-VL-1.5-8B-auto":partial(vlm.KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B"),
    "Keye-VL-1.5-8B-think":partial(vlm.KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B", think=True),
    "Keye-VL-1.5-8B-nothink":partial(vlm.KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B", no_think=True), 
    "Keye-VL-8B-Preview-think":partial(vlm.KeyeChat, model_path="Kwai-Keye/Keye-VL-8B-Preview", think=True), 
}

qianfanvl_series = {
    'Qianfan-VL-3B': partial(vlm.Qianfan_VL, model_path='baidu/Qianfan-VL-3B'),
    'Qianfan-VL-8B': partial(vlm.Qianfan_VL, model_path='baidu/Qianfan-VL-8B'),
    'Qianfan-VL-70B': partial(vlm.Qianfan_VL, model_path='baidu/Qianfan-VL-70B'),
}

lfm2vl_series = {
    "LFM2-VL-450M": partial(vlm.LFM2VL, model_path="LiquidAI/LFM2-VL-450M"),
    "LFM2-VL-1.6B": partial(vlm.LFM2VL, model_path="LiquidAI/LFM2-VL-1.6B"),
    "LFM2-VL-3B": partial(vlm.LFM2VL, model_path="LiquidAI/LFM2-VL-3B"),
    "LFM2.5-VL-1.6B": partial(vlm.LFM2VL, model_path="LiquidAI/LFM2.5-VL-1.6B"),
}

covt_series = {
    "CoVT-7B-seg": partial(
        vlm.CoVTChat,
        model_path="Wakals/CoVT-7B-seg",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "CoVT-7B-depth": partial(
        vlm.CoVTChat,
        model_path="Wakals/CoVT-7B-depth",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "CoVT-7B-seg_depth_dino": partial(
        vlm.CoVTChat,
        model_path="Wakals/CoVT-7B-seg_depth_dino",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "CoVT-7B-seg_depth_dino_edge": partial(
        vlm.CoVTChat,
        model_path="Wakals/CoVT-7B-seg_depth_dino_edge",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
}

bagel_series = {
    "BAGEL-7B-MoT": partial(vlm.Bagel, model_path='ByteDance-Seed/BAGEL-7B-MoT'),
}

spatial_related_models = {
    # 3B models
    "MindCube-Qwen2.5VL-RawQA-SFT": partial(
        vlm.Qwen2VLChat,
        model_path="MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MindCube-Qwen2.5VL-Aug-CGMap-FFR-Out-SFT": partial(
        vlm.Qwen2VLChat,
        model_path="MLL-Lab/MindCube-Qwen2.5VL-Aug-CGMap-FFR-Out-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MindCube-Qwen2.5VL-Plain-CGMap-FFR-Out-SFT": partial(
        vlm.Qwen2VLChat,
        model_path="MLL-Lab/MindCube-Qwen2.5VL-Plain-CGMap-FFR-Out-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "SpatialLadder-3B": partial(
        vlm.Qwen2VLChat,
        model_path="hongxingli/SpatialLadder-3B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Spatial-MLLM-subset-sft": partial(
        vlm.SpatialMLLM,
        model_path="Diankun/Spatial-MLLM-subset-sft",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_num_frames=16,
        use_custom_prompt=False,
        post_process=True,
    ),
    "VST-3B-SFT": partial(
        vlm.Qwen2VLChat,
        model_path="rayruiyang/VST-3B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    
    # 7B models
    "SpaceR-SFT-7B": partial(
        vlm.Qwen2VLChat,
        model_path="RUBBISHLIKE/SpaceR-SFT-7B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "ViLaSR": partial(
        vlm.Qwen2VLChat,
        model_path="inclusionAI/ViLaSR",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "VST-7B-SFT": partial(
        vlm.Qwen2VLChat,
        model_path="rayruiyang/VST-7B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "VLM-3R": partial(
        vlm.VLM3R, 
        model_path="Journey9ni/vlm-3r-llava-qwen2-lora",
    ),
}

sensenova_si_series = {
    # SenseNova-SI-1.0 series
    "SenseNova-SI-InternVL3-2B": partial(
        vlm.InternVLChat, 
        model_path="sensenova/SenseNova-SI-InternVL3-2B", 
        use_custom_prompt=False,
        version="V2.0"
    ),
    "SenseNova-SI-InternVL3-8B": partial(
        vlm.InternVLChat, 
        model_path="sensenova/SenseNova-SI-InternVL3-8B", 
        use_custom_prompt=False,
        version="V2.0"
    ),
    # SenseNova-SI-1.1 series
    "SenseNova-SI-1.1-InternVL3-2B": partial(
        vlm.InternVLChat, 
        model_path="sensenova/SenseNova-SI-1.1-InternVL3-2B", 
        use_custom_prompt=False,
        version="V2.0"
    ),
    "SenseNova-SI-1.1-InternVL3-8B": partial(
        vlm.InternVLChat, 
        model_path="sensenova/SenseNova-SI-1.1-InternVL3-8B", 
        use_custom_prompt=False,
        version="V2.0"
    ),
    "SenseNova-SI-1.1-Qwen2.5-VL-3B": partial(
        vlm.Qwen2VLChat, 
        model_path="sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B", 
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "SenseNova-SI-1.1-Qwen2.5-VL-7B": partial(
        vlm.Qwen2VLChat, 
        model_path="sensenova/SenseNova-SI-1.1-Qwen2.5-VL-7B", 
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "SenseNova-SI-1.1-Qwen3-VL-8B": partial(
        vlm.Qwen3VLChat,
        model_path="sensenova/SenseNova-SI-1.1-Qwen3-VL-8B",
        use_custom_prompt=False,
    ),
    "SenseNova-SI-1.1-BAGEL-7B-MoT": partial(
        vlm.Bagel, 
        model_path='sensenova/SenseNova-SI-1.1-BAGEL-7B-MoT'
    ),
    # SenseNova-SI-1.2 series
    "SenseNova-SI-1.2-InternVL3-8B": partial(
        vlm.InternVLChat, 
        model_path="sensenova/SenseNova-SI-1.2-InternVL3-8B", 
        use_custom_prompt=False,
        version="V2.0"
    ),
    # SenseNova-SI-1.3 series
    "SenseNova-SI-1.3-InternVL3-8B": partial(
        vlm.InternVLChat, 
        model_path="sensenova/SenseNova-SI-1.3-InternVL3-8B", 
        use_custom_prompt=False,
        version="V2.0"
    ),
}

internvl_groups = [
    internvl, internvl2, internvl2_5, mini_internvl, internvl2_5_mpo, 
    internvl3, internvl3_5
]
internvl_series = {}
for group in internvl_groups:
    internvl_series.update(group)

interns1_groups = [
    interns1_mini
]
interns1_series = {}
for group in interns1_groups:
    interns1_series.update(group)
    
supported_VLM = {}

model_groups = [
    ungrouped, o1_apis, api_models, xtuner_series, qwen_series, llava_series, granite_vision_series,
    internvl_series, yivl_series, xcomposer_series, minigpt4_series, 
    idefics_series, instructblip_series, deepseekvl_series, deepseekvl2_series, deepseekocr_series,
    janus_series, minicpm_series, cogvlm_series, wemm_series, cambrian_series, 
    chameleon_series, video_models, ovis_series, vila_series, mantis_series,
    mmalaya_series, phi3_series, phi4_series, xgen_mm_series, qwen2vl_series, qwen3vl_series, qwen3_5_series,
    slime_series, eagle_series, moondream_series, llama_series, molmo_series,
    kosmos_series, points_series, nvlm_series, vintern_series, h2ovl_series,
    aria_series, smolvlm_series, sail_series, valley_series, vita_series,
    ross_series, emu_series, ola_series, ursa_series, gemma_series,
    long_vita_series, ristretto_series, kimi_series, aguvis_series, hawkvl_series,
    flash_vl, kimi_vllm_series, oryx_series, treevgr_series, varco_vision_series, qtunevl_series, 
    xvl_series, thyme_series, logics_series, cosmos_series, keye_series, qianfanvl_series, 
    lfm2vl_series, rbdashmm_api_series_lmdeploy, interns1_series, insight_v_series, covt_series
]

# add by EASI team
model_groups.extend([bagel_series, spatial_related_models, sensenova_si_series])

for grp in model_groups:
    supported_VLM.update(grp)
