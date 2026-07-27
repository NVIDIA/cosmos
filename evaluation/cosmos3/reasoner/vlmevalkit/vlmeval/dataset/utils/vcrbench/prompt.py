# flake8: noqa
Recall_Evaluation_Prompt = """You are an expert system for verifying solutions to video-based problems. Your task is to match the ground truth middle steps with the provided solution.

INPUT FORMAT:
1. Problem: The original question/task
2. A Solution of a model
3. Ground Truth: Essential steps required for a correct answer

MATCHING PROCESS:

You need to match each ground truth middle step with the solution:

Match Criteria:
- The middle step should exactly match in the content or is directly entailed by a certain content in the solution
- All the details must be matched, including the specific value and content
- You should judge all the middle steps for whethere there is a match in the solution

Step Types:
1. Logical Inference Steps
   - Contains exactly one logical deduction
   - Must produce a new derived conclusion
   - Cannot be just a summary or observation

2. Video Description Steps
   - Pure visual observations
   - Only includes directly visible elements
   - No inferences or assumptions
   - Contains event time

3. Background Review Steps:
   - Repetition or review of the problem
   - Not directly related to solving the problem.

OUTPUT FORMAT:
JSON array of judgments:
[
  {{
    "step": ground truth middle step,
    "step_type": "Video Description Steps|Logical Inference Steps|Background Review Steps",
    "judgment": "Matched" | "Unmatched",
  }}
]

ADDITIONAL RULES:
1. Only output the json array with no additional information.
2. Judge each ground truth middle step in order without omitting any step.

Here is the problem, answer, solution, and the ground truth middle steps:
"""

Precision_Evaluation_Prompt = """
# Task Overview
Given a solution with multiple reasoning steps for an video-based problem, reformat it into well-structured steps and evaluate their correctness.

# Step 1: Reformatting the Solution
Convert the unstructured solution into distinct reasoning steps while:
- Preserving all original content and order
- Not adding new interpretations
- Not omitting any steps

## Step Types
1. Logical Inference Steps
   - Contains exactly one logical deduction
   - Must produce a new derived conclusion
   - Cannot be just a summary or observation

2. Video Description Steps
   - Pure visual observations
   - Only includes directly visible elements
   - No inferences or assumptions
   - Contains event time

3. Background Review Steps:
   - Repetition or review of the problem
   - Not directly related to solving the problem.

## Step Requirements
- Each step must be atomic (one conclusion per step)
- No content duplication across steps
- Initial analysis counts as background information
- Final answer determination counts as logical inference

# Step 2: Evaluating Correctness
Evaluate each step against:

## Ground Truth Matching
For video descriptions:
- Key elements must match ground truth descriptions

For logical inferences:
- Conclusion must EXACTLY match or be DIRECTLY entailed by ground truth

For Background review:
-  Without special circumstances are deemed to be redundant

## Reasonableness Check (if no direct match)
If Step:
- Premises must not contradict any ground truth or correct answer
- Logic is valid
- Conclusion must not contradict any ground truth
- Conclusion must support or be neutral to correct answer
- Helpful in solving the problem, non-redundant steps
this Step be viewed as matched.

## Judgement Categories
- "Match": Aligns with ground truth
- "Wrong": Contradictory with ground truth
- "Redundant": Redundant steps that do not help solve the problem

# Output Requirements
1. The output format MUST be in valid JSON format without ANY other content.
2. For highly repetitive patterns, output it as a single step.
3. Output maximum 35 steps. Always include the final step that contains the answer.

Here is the json output format:
## Output Format
[
  {{
    "step": "reformatted the solution step",
    "step_type": "Video Description Steps|Logical Inference Steps|Background Review Steps",
    "reasons_for_judgment": "The reason for judging the matching result of the step in the solution based on Ground Truth Information. Sufficient evidence needs to be found in Ground Truth Information to determine the correctness of the reformatted the solution step. The video event description time error is no more than 3 seconds and is considered correct. If the solution step does not specify the time, it is considered wrong.",
    "judgment": "Matched|Wrong|Redundant",
  }}
]

Here is the problem, and the solution that needs to be reformatted to steps:

"""

Answer_Extraction_Prompt_part1 = """You are an AI assistant who will help me to extract an answer of a question. You are provided with a question and a response, and you need to find the final answer of the question.

Extract Rule:
[Multiple choice question]
1. The answer could be answering the option letter or the value. You should directly output the choice letter of the answer.
2. You should output a single uppercase character in A, B, C, D, E, F, G, H, I (if they are valid options), and Z.
3. If the meaning of all options are significantly different from the final answer, output Z.
[Non Multiple choice question]
1. Output the final value of the answer. It could be hidden inside the last step of calculation or inference. Pay attention to what the question is asking for to extract the value of the answer.
2. The final answer could also be a short phrase or sentence.
3. If the answer is about a certain time period, such as from 1 minute 30 seconds to 2 minutes 30 seconds, it should be given in the format [90, 150].
4. If the response doesn't give a final answer, output Z.

Output Format:
Directly output the extracted answer of the response

Example 1:
Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog
Response: a cute teddy bear
Your output: A
Example 2:
Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog
Answer: Spider
Your output: Z
Example 3:
Question: On a flat playground, choose a Cartesian Oxy coordinate system (in unit of meters). A child running at a constant velocity $V=1 \\mathrm{~m} / \\mathrm{s}$ around a heart-shaped path satisfies the following order- 6 algebraic equation:\n$$\n\\left(x^{2}+y^{2}-L^{2}\\right)^{3}-L x^{2} y^{3}=0, L=10 \\text {. }\n$$\nWhen the child is at the position $(x, y)=(L, 0)$, what is the magnitude of their acceleration?\n\n<image>
Response: fore, \\(r \\approx 10 \\, \\text{m}\\).\n\n7. **Calculating Acceleration**: Using the radius of curvature:\n   \\[\n   a_c = \\frac{V^2}{r} = \\frac{(1 \\, \\text{m/s})^2}{10 \\, \\text{m}} = 0.1 \\, \\text{m/s}^2\n   \\]\n\n8. **Magnitude of Acceleration**: The magnitude of the acceleration is the sum of the tangential and centripetal accelerations. Since the tangential acceleration is zero, the magnitude of the acceleration is purely centripetal:\n   \\[\n   \\text{Magnitude of acceleration} = a_c = 0.1 \\, \\text{m/s}^2\n   \\]\n\nAnswer: \\boxed{0.1 \\, \\text{m/s}^2}
Your output: 0.1 text{m/s}^2
Example 4:
Question: In the diagram, $O A=15, O P=9$ and $P B=4$. Determine the equation of the line through $A$ and $B$. Explain how you got your answer.

<image>
Response: The equation of the line through $A$ and $B$ is $y = -\frac{1}{3}x + 12$.

To find the equation of the line through $A$ and $B$, we first need to find the slope of the line. The slope of a line passing through two points $(x_1, y_1)$ and $(x_2, y_2)$ is given by the formula:

$$\text{Slope} = \frac{y_2 - y_1}{x_2 - x_1}$$

Using the coordinates of points $A(0, 15)$ and $B(4, 0)$, we can calculate the slope as follows:

$$\text{Slope} = \frac{0 - 15}{4 - 0} = \frac{-15}{4} = -\frac{15}{4}$$

Now that we have the slope, we can use the point-slope form of a linear equation to find the equation of the line. The point-slope form is given by:

$$y - y_1 = m(x - x_1)$$

where $(x_1, y_1)$ is a point on the line and $m$ is the slope. In this case, we can use point $A(0, 15)$ and the slope $-\frac{15}{4}$:

$$y - 15 = -\frac{15}{4}(x - 0)$$

Simplifying the equation, we get:

$$y - 15 = -\frac{15}{4}x$$

$$y = -\frac{15}{4}x + 15$$

Therefore, the equation of the line through $A$ and $B$ is $y = -\frac{15}{4}x + 15$.
Your output: $y = -\frac{15}{4}x + 15$
Example 5:
Question: <image> <image> <image> <image> <image> <image> <image> <image> <image>In the paper \'An Object is Worth 64x64 Pixels: Generating 3D Object via Image Diffusion\', what aspect of the proposed method does this figure aim to demonstrate?
Response: the potential of this 3D generation paradigm to generate objects with PBR materials
Your output: the potential of this 3D generation paradigm to generate objects with PBR materials
Example 6:
Question: Please specify the start and end time of the event 'An intro screen appears with a gray background and the red and white words that say PRO TRAINING.'
Response: The start and end time of the event 'An intro screen appears with a gray background and the red and white words that say PRO TRAINING.' are as follows:\n\n**Step 1: Identify the start time of the event.**\nThe video starts at 00:00. The intro screen is visible from the very beginning.\n\n**Start time:** 00:00\n\n**Step 2: Identify the end time of the event.**\nThe intro screen starts to disappear and transition to the next scene at 00:11. By 00:12, the intro screen is completely gone. So, the intro screen is visible until 00:11.\n\n**End time:** 00:11\n\nTherefore, the event 'An intro screen appears with a gray background and the red and white words that say PRO TRAINING' starts at **00:00** and ends at **00:11**.\n\n**Answer:**\nStart time: 00:00\nEnd time: 00:11
Your output: [0,11]

"""

Answer_Extraction_Prompt_part2 = """
Question: {question}
Answer: {response}
Your output:
"""

Answer_Scoring_Prompt_part1 = r"""You are an AI assistant who will help me to judge whether two answers are consistent.

Input Illustration:
[Standard Answer] is the standard answer to the question
[Model Answer] is the answer extracted from a model's output to this question.

Task Illustration:
Determine whether [Standard Answer] and [Model Answer] are consistent.
Consistent Criteria:
[Multiple-Choice questions]
1. If the [Model Answer] is the option letter, then it must completely matches the [Standard Answer].
2. If the [Model Answer] is not an option letter, then the [Model Answer] must completely match the option content of [Standard Answer].
[Nan-Multiple-Choice questions]
1. The [Model Answer] and [Standard Answer] should exactly match.
2. If the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.

Output Format:
1. If they are consistent, output 1; if they are different, output 0.
2. DIRECTLY output 1 or 0 without any other content.

Example 1:
Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog
[Model Answer]: a cute teddy bear
[Standard Answer]: A
Your output: 1

Example 2:
Question: Find the value of AB. Choices: A.1;B.5;C.9;D.10
[Model Answer]: \\boxed{5}
[Standard Answer]: B
Your output: 1

Example 3:
Question: Three of the following four slides are from the same presentation, but one is from a different one. Please identify the outlier: \n\n<image> <image> <image> <image>\nA. the forth image\nB. the second image\nC. the third image\nD. None of the choices provided
[Model Answer]: \\boxed{B}
[Standard Answer]: A
Your output: 0


"""

Answer_Scoring_Prompt_part2 = """
Question: {question}
[Model Answer]: {extract_answer}
[Standard Answer]: {gt_answer}
Your output:
"""


def build_Extraction_prompt(item):
    tmpl = 'Question: {question}\nAnswer: {response}\nYour output:'
    return tmpl.format(question=item['question'], response=item['prediction'])


def build_Scoring_prompt(item):
    tmpl = 'Question: {question}\n[Model Answer]: {extract_answer}\n[Standard Answer]: {gt_answer}\nYour output:'
    return tmpl.format(question=item['question'], extract_answer=item['extracted_answer'], gt_answer=item['answer'])


def build_Precision_prompt(item):
    tmpl = '[Problem]:{question}\n[Solution]:{solution}\n[Ground Truth Information]:{gt_annotation}'
    return tmpl.format(question=item['question'], solution=item['prediction'], gt_annotation=item['reasoning'])


def build_Recall_prompt(item):
    tmpl = '[Problem]:{question}\n[Answer]:{answer}\n[Solution]:{solution}\n[Ground Truth Information]:{gt_annotation}'
    return tmpl.format(question=item['question'], answer=item['answer'], solution=item['prediction'], gt_annotation=item['reasoning'])
