"""
Local wrapper for nvidia/Cosmos-Reason1-7B model.

This model is a video-language model based on Qwen2-VL architecture.
It can be run locally without API keys.
"""

import copy

import torch

from .base import BaseModel


class CosmosReasonHF(BaseModel):
    """
    Local implementation of nvidia/Cosmos-Reason1-7B video language model.

    This model processes videos directly (not frames) and matches the format
    used in the original cosmos-reason1 project.

    Args:
        model_path: Path to model (default: 'nvidia/Cosmos-Reason1-7B')
        fps: Frames per second for video processing (default: 16.0)
        nframes: Number of frames to extract (default: None - use fps)
        total_pixels: Total pixels for video processing (default: 8192 * 28 * 28)
        max_pixels: Maximum pixels per frame (default: None)
        min_pixels: Minimum pixels per frame (default: None)
        max_frames: Maximum number of frames (default: None)
        max_new_tokens: Maximum tokens to generate (default: 4096)
        temperature: Sampling temperature (default: 0.6)
        top_p: Nucleus sampling parameter (default: 0.95)
        top_k: Top-k sampling parameter (default: 50)
        repetition_penalty: Repetition penalty (default: 1.05)
        use_vllm: Use vLLM for inference (default: False)
        verbose: Print debug information (default: False)

    Note:
        Dataset-provided video processing parameters (fps, total_pixels, etc.)
        will override the model's default parameters when provided.
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True  # This model supports video input directly

    def __init__(
        self,
        model_path='nvidia/Cosmos-Reason1-7B',
        # Using default parameters from cosmos-reason1 examples:
        # https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/video_critic/video_critic.py
        fps=16.0,
        nframes=None,
        total_pixels=8192 * 28 * 28,
        max_pixels=None,
        min_pixels=None,
        max_frames=None,
        image_patch_size=14,
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.05,
        use_vllm=False,
        tensor_parallel_size=1,
        verbose=False,
        **kwargs
    ):
        super().__init__()

        self.model_path = model_path
        self.fps = fps
        self.nframes = nframes
        self.total_pixels = total_pixels
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_frames = max_frames
        self.image_patch_size = image_patch_size
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.verbose = verbose

        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        if self.verbose:
            print(f"Loading Cosmos-Reason1 model from {model_path}")
            print("Model's video settings:")
            print(f"  - fps: {fps}")
            print(f"  - nframes: {nframes}")
            print(f"  - total_pixels: {total_pixels}")
            print(f"  - max_pixels: {max_pixels}")
            print(f"  - min_pixels: {min_pixels}")
            print(f"  - max_frames: {max_frames}")
            print(f"Using vLLM: {use_vllm}")

        if use_vllm:
            # Use vLLM for inference
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=model_path,
                enforce_eager=True,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.9,
            )

            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_new_tokens,
                seed=1,  # for reproducibility
            )

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            if self.verbose:
                print("vLLM model loaded successfully")
        else:
            # Use standard HuggingFace transformers
            from transformers import AutoModelForImageTextToText, AutoProcessor

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # Load model with language modeling head for generation (supports both images and videos)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            self.model.eval()
            torch.cuda.empty_cache()

            if self.verbose:
                print(f"✓ Model loaded successfully on device: {self.model.device}")

    def _get_video_processing_kwargs(self, msg) -> dict:
        """
        Extract video processing kwargs from message, with dataset overrides.

        If the message contains video processing parameters (fps, total_pixels, etc.),
        they take precedence over the model's default parameters.

        Args:
            msg: Message dict that may contain video processing kwargs

        Returns:
            dict: Video processing kwargs to use
        """
        process_video_kwargs = copy.deepcopy(msg)

        # Remove non-video-processing keys
        process_video_kwargs.pop('type', None)
        process_video_kwargs.pop('value', None)

        # List of valid video processing parameter names
        video_processing_kwargs_keys = ['nframes', 'fps', 'total_pixels', 'max_pixels', 'min_pixels', 'max_frames']

        # If the message already has video processing kwargs, use them (dataset override)
        if any(k in process_video_kwargs for k in video_processing_kwargs_keys):
            if self.verbose:
                print(f"Using dataset-provided video processing kwargs: {process_video_kwargs}")
            return process_video_kwargs

        # Otherwise, use model's default parameters
        for video_processing_kwargs_key in video_processing_kwargs_keys:
            if video_processing_kwargs_value := getattr(self, video_processing_kwargs_key, None):
                process_video_kwargs[video_processing_kwargs_key] = video_processing_kwargs_value

        if self.verbose:
            print(f"Using model default video processing kwargs: {process_video_kwargs}")

        return process_video_kwargs

    def generate_inner(self, message, dataset=None):
        """
        Generate response for given message.

        Message format from VideoPhy2:
        [
            {"type": "text", "value": SYSTEM_PROMPT, "role": "system"},
            {"type": "video", "value": video_path},
            {"type": "text", "value": question}
        ]

        Args:
            message: List of message dicts
            dataset: Dataset name (optional)

        Returns:
            str: Generated text response
        """
        from qwen_vl_utils import process_vision_info

        # Parse message to extract system, video/image, and user text
        system_prompt = ""
        video_path = None
        video_msg = None  # Store the full video message to extract kwargs
        image_paths = []
        user_text = ""

        for msg in message:
            if msg.get('role') == 'system':
                system_prompt = msg.get('value', '')
            elif msg['type'] == 'video':
                video_path = msg['value']
                video_msg = msg  # Store full message for kwargs extraction
            elif msg['type'] == 'image':
                image_paths.append(msg['value'])
            elif msg['type'] == 'text' and msg.get('role') != 'system':
                user_text = msg['value']

        # Build messages in the format expected by the model
        messages = []

        # Add system message if present
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user message with video/images and text
        user_content = []

        if video_path:
            # Extract video processing kwargs from the message (with dataset overrides)
            video_kwargs = self._get_video_processing_kwargs(video_msg)

            user_content.append({
                "type": "video",
                "video": video_path,
                **video_kwargs  # Use extracted kwargs (dataset overrides or model defaults)
            })
        elif image_paths:
            # Image input(s)
            for image_path in image_paths:
                user_content.append({
                    "type": "image",
                    "image": image_path,
                })
        else:
            raise ValueError("No video or image found in message")

        # Add text prompt
        user_content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": user_content})

        if self.verbose:
            if video_path:
                print(f"Processing video: {video_path}")
            elif image_paths:
                print(f"Processing {len(image_paths)} image(s): {image_paths[0]}")
            print(f"User question: {user_text[:100] if len(user_text) > 100 else user_text}")

        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision info (videos/images)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=self.image_patch_size
        )

        if self.use_vllm:
            # vLLM inference path
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": text_prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }

            if self.verbose:
                print(f"Generating with vLLM: temperature={self.sampling_params.temperature}, "
                      f"max_tokens={self.sampling_params.max_tokens}")

            # Generate with vLLM
            outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
            response = outputs[0].outputs[0].text

        else:
            # Standard transformers inference path
            # Prepare inputs for model
            model_inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )

            # Move to device
            model_inputs = model_inputs.to(self.model.device)

            if self.verbose:
                print(f"Input shape: {model_inputs.input_ids.shape}")
                print(f"Generating with: temperature={self.generate_kwargs['temperature']}, "
                      f"max_new_tokens={self.generate_kwargs['max_new_tokens']}")

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **model_inputs,
                    **self.generate_kwargs
                )

            # Decode output
            # Remove input tokens from output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
            ]

            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return response

    def generate(self, message, dataset=None):
        """
        Main generate method called by the framework.

        Args:
            message: List of message dicts or string
            dataset: Dataset name for custom handling

        Returns:
            str: Generated response
        """
        if isinstance(message, str):
            message = [dict(type='text', value=message)]

        return self.generate_inner(message, dataset=dataset)


# Test function
def test_cosmos_reason_hf(use_vllm=False):
    """Test the CosmosReasonHF model with video and image inputs.

    Args:
        use_vllm: If True, test with vLLM backend. If False, test with standard transformers.
    """
    import os

    backend_name = "vLLM" if use_vllm else "Transformers"
    print("=" * 60)
    print(f"Testing CosmosReasonHF ({backend_name})")
    print("=" * 60)

    # Initialize the model
    print(f"\n[1/4] Initializing model with {backend_name}...")
    print("Note: This will download the model (~14GB) if not already cached")
    try:
        model = CosmosReasonHF(
            model_path='nvidia/Cosmos-Reason1-7B',
            use_vllm=use_vllm,
            verbose=True
        )
        print(f"✓ Model initialized successfully with {backend_name}")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get root directory
    # Go up three levels from vlmeval/vlm/ to get to the root
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Test 1: Video input
    print("\n[2/4] Testing video input...")
    video_path = os.path.join(root_dir, "assets", "sample.mp4")

    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        return False

    print(f"Video path: {video_path}")

    # Build message for video
    video_message = [
        {"type": "video", "value": video_path},
        {"type": "text", "value": "Describe the video."}
    ]

    print(f"Message format: {len(video_message)} parts")
    print(f"  - Part 1: video ({os.path.basename(video_path)})")
    print(f"  - Part 2: text ('{video_message[1]['value']}')")

    # Generate response for video
    print(f"\nGenerating response for video with {backend_name}...")
    print("-" * 60)
    try:
        response = model.generate(video_message, dataset=None)
        print("✓ Video response generated successfully")
        print("-" * 60)
        print(f"\nVideo Response ({backend_name}):")
        print(response)
        print("-" * 60)
    except Exception as e:
        print(f"✗ Failed to generate video response: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Image input
    print("\n[3/4] Testing image input...")
    image_path = os.path.join(root_dir, "assets", "apple.jpg")

    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False

    print(f"Image path: {image_path}")

    # Build message for image
    image_message = [
        {"type": "image", "value": image_path},
        {"type": "text", "value": "Describe the image."}
    ]

    print(f"Message format: {len(image_message)} parts")
    print(f"  - Part 1: image ({os.path.basename(image_path)})")
    print(f"  - Part 2: text ('{image_message[1]['value']}')")

    # Generate response for image
    print(f"\n[4/4] Generating response for image with {backend_name}...")
    print("-" * 60)
    try:
        response = model.generate(image_message, dataset=None)
        print("✓ Image response generated successfully")
        print("-" * 60)
        print(f"\nImage Response ({backend_name}):")
        print(response)
        print("-" * 60)
        return True
    except Exception as e:
        print(f"✗ Failed to generate image response: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test CosmosReasonHF model')
    parser.add_argument('--use-vllm', action='store_true', help='Use vLLM backend for inference')
    args = parser.parse_args()

    success = test_cosmos_reason_hf(use_vllm=args.use_vllm)
    exit(0 if success else 1)
