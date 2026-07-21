SYSTEM_PROMPTS_EN = """You are participating in a high school physics competition.
Please read the following question carefully and provide a clear, step-by-step solution with full reasoning.

Context (if applicable):
{context}

Problem:
{problem}

Useful information (formulas, constants, units, if applicable):
{information}

Instructions:
1. Use LaTeX to format all variables, equations, and calculations.
2. Enclose your full reasoning process within <think> </think> tags.
3. Provide the final answer within <answer> </answer> tags, using the format
   of [\\boxed{{answer}}]. Do not include units inside the box.
4. For multiple sub-questions, list the answers in order using the format:
   [\\boxed{{answer1}}, \\boxed{{answer2}}, ...].
5. For multiple-choice questions, provide the final selected option(s) in the
   boxed answer instead of the calculation result (e.g., [\\boxed{{A}}]).

Example of Output:
<think>
Step 1: Analyze the problem... Step 2: Apply the relevant equations...
</think>
<answer>
[\\boxed{{A}}, \\boxed{{3.2}}]
</answer>
"""

SYSTEM_PROMPTS_ZH = """你正在参加高中物理竞赛。
请仔细阅读下列题目，结合上下文信息，详细推导并给出清晰、有条理的解题步骤与完整的逻辑推理过程。

背景信息（如有）：
{context}

题目内容：
{problem}

可用信息（如物理公式、常数、单位等）：
{information}

作答要求：
1. 所有物理量、公式和计算过程须使用 LaTeX 格式书写。
2. 将完整的推理过程用 <think> 和 </think> 标签括起来。
3. 将最终答案置于 <answer> 和 </answer> 标签中，答案格式为 [\\boxed{{答案}}]，方框内不包含单位。
4. 对于包含多个小问的题目，按顺序列出所有答案，格式为：[\\boxed{{答案1}}, \\boxed{{答案2}}, ...]。
5. 对于选择题，请在答案的方框中给出最终选择的选项，而不是计算结果（例如：[\\boxed{{A}}]）。

输出示例:
<think>
第一步: 分析问题... 第二步: 运用相关公式...
</think>
<answer>
[\\boxed{{A}}, \\boxed{3.2}]
</answer>
"""

# Judge grading prompt template
JUDGE_GRADING_PROMPT_TEMPLATE = """You are an expert physics competition grader.
Evaluate the student's solution against the specific grading criterion.

PHYSICS PROBLEM:
{question}

STUDENT'S SOLUTION:
{prediction}

GRADING CRITERION:
{criterion_description}{total_score_warning}{retry_warning}

INSTRUCTIONS:
1. Carefully analyze the student's solution for physics concepts, mathematical derivations, and calculations.
2. Compare the solution against the specific grading criterion provided.
3. Award points strictly according to the criterion, including partial credit when specified.
4. Consider both conceptual understanding and technical accuracy.
5. BE CONSERVATIVE - remember this is one of multiple criteria being evaluated simultaneously.

SCORING FORMAT:
- Read the grading criterion carefully to understand the maximum points and conditions for partial credit
- Evaluate whether the student's solution meets the full criteria, partial criteria, or no criteria
- Output your score using the exact format: \\boxed{{score}}
- The score should be a number (e.g., 0.4, 0.2, 0.1, 0.0)

CRITICAL REQUIREMENTS:
- You MUST output your final score in the format: \\boxed{{score}}
- The score must be a single number only (no text inside the boxed)
- Do not include explanations after the boxed score
- Ensure your score follows the point allocation in the grading criterion
- BE CONSERVATIVE to avoid exceeding the total score limit

Example outputs:
- \\boxed{{0.4}} (for full credit)
- \\boxed{{0.1}} (for partial credit)
- \\boxed{{0.0}} (for no credit)

⚠️ CRITICAL INSTRUCTION:
- Output ONLY: \\boxed{{score}}
- NO explanations, NO analysis, NO reasoning
- Just the number in the exact format \\boxed{{score}}
- Any other text will result in AUTOMATIC REJECTION

RESPOND WITH ONLY THE BOXED SCORE:"""

# Total score limit warning template
TOTAL_SCORE_WARNING_TEMPLATE = """
⚠️  IMPORTANT TOTAL SCORE CONSTRAINT:
- This question has a maximum total score of {max_total_score} points
- ALL marking criteria scores combined MUST NOT exceed {max_total_score} points
- You are evaluating ONE criterion among multiple criteria for this question
- Be conservative in your scoring to ensure the total doesn't exceed the limit
- This is attempt #{current_attempt} of evaluation"""

# Retry warning template
RETRY_WARNING_TEMPLATE = """
🔄 RETRY NOTICE:
- Previous attempt(s) resulted in total score exceeding the maximum
- Please be more conservative in your scoring
- Focus on strict adherence to the criterion requirements"""
