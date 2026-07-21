from PIL import Image

REASONING_SYS_PROMPT = (
    "You are a helpful assistant.\n\n"
    "Solve the following problem step by step, and optionally write Python code "
    "for image manipulation to enhance your reasoning process. The Python code "
    "will be executed by an external sandbox, and the processed image or result "
    "(wrapped in <sandbox_output></sandbox_output>) can be returned to aid your "
    "reasoning and help you arrive at the final answer.\n\n"
    "**Reasoning & Image Manipulation (Optional but Encouraged):**\n"
    "    * You have the capability to write executable Python code to perform "
    "image manipulations (e.g., cropping to a Region of Interest (ROI), "
    "resizing, rotation, adjusting contrast) or perform calculation for better "
    "reasoning.\n"
    "    * The code will be executed in a secure sandbox, and its output will be "
    "provided back to you for further analysis.\n"
    "    * All Python code snippets **must** be wrapped as follows:\n"
    "    <code>\n"
    "    ```python\n"
    "    # your code.\n"
    "    ```\n"
    "    </code>\n"
    "    * At the end of the code, print the path of the processed image "
    "(processed_path) or the result for further processing in a sandbox "
    "environment."
)


SIMPLE_SYS_PROMPT = "You are a helpful assistant."


def generate_prompt_simple_qa(user_question):
    # Construct the prompt based on the given requirements
    prompt = (
        "You are an advanced AI assistant specializing in visual question "
        "answering (VQA). You don't need to perform any image manipulation "
        "or reasoning. Give the answer to the following question directly.\n"
        f"**User's Question:** \"{user_question}\""
    )
    return prompt


def generate_prompt_final_qa(user_question, user_image_path):
    # Construct the prompt based on the given requirements
    try:
        with Image.open(user_image_path) as img:
            user_image_size = f"{img.width}x{img.height}"
    except Exception:
        user_image_size = "Unable to determine (error reading image)"

    prompt = f"""<image>
{user_question}

### User Image Path:** "{user_image_path}"
### User Image Size:** "{user_image_size}"

### **Output Format (strict adherence required):**

<think>Your detailed reasoning process, including any code, should go here.</think>
<answer>Your final answer to the user's question goes here.</answer>
"""
    return prompt


SPECIAL_STRING_LIST = ["</code>", "</answer>"]
