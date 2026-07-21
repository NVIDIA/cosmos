import json
import re

import pandas as pd
import sympy

from vlmeval.smp import load

FAIL_MSG = 'Failed to obtain answer via API.'

id_answer_pair_prompt = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
Example:
    Question: Each cycle represents a number. You need to find out what the three numbers are. Give a possible answer in the format 'cycle id:number'
    Model Response: The possible answer is: A:5, B:1, C:2
    Extracted answer (json format):
        {{
            "A":5,
            "B":1,
            "C":2
        }}

Please extract the answer for the following response:
    Question: {question}
    Model Response: {response}
    You should only output the json without any other texts.
"""

# id: number, answer: letter
id_answer_pair_prompt_reverse = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
Example:
    Question: Each cycle (id from 1 to 3) represents a letter. You need to find out what the three numbers are. Give a possible answer in the format 'cycle id:letter'
    Model Response: The possible answer is: 1:A, 2:B, 3:C
    Extracted answer (json format):
        {{
            "1":"A",
            "2":"B",
            "3":"C"
        }}

Please extract the answer for the following response:
    Question: {question}
    Model Response: {response}
    You should only output the json without any other texts.
"""

coordinate_answer_prompt = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
Example1:
    Question: According to the clues, find the corresponding position. Answer in '(row id (A-C), column id (1-3))' format.
    Model Response: The possible answer is: (A, 1)
    Extracted answer (json format):
        [
            {{
                "row": "A",
                "column": 1
            }}
        ]
Example2:
    Question: According to the clues, find the two corresponding position. Answer in '(row id (A-C), column id (1-3))' format.
    Model Response: The possible answer is: (A, 1), (B, 3)
    Extracted answer (json format):
        [
            {{
                "row": "A",
                "column": 1
            }},
            {{
                "row": "B",
                "column": 3
            }}
        ]

Please extract the answer for the following response:
    Question: {question}
    Model Response: {response}
    You should only output the json without any other texts.
"""

multiple_answer_prompt = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
Example1:
    Question: There are six balls (A-F), find two balls that have lower weight. Answer in ball id.
    Model Response: The possible answer is: A, C
    Extracted answer (json format):
        {{
            "id": ["A", "C"]
        }}
Example2:
    Question: There are six balls (A-F), find a ball that have lower weight. Answer in ball id.
    Model Response: The possible answer is: A
    Extracted answer (json format):
        {{
            "id": ["A"]
        }}
Please extract the answer for the following response:
    Question: {question}
    Model Response: {response}
    You should only output the json without any other texts.
"""

formula_extraction_prompt = """Please extract the mathematical formula from the following model response and type it at the end of the prompt.
Example:
    Question: What is the right equation to solve the problem?
    Model Response: The right equation to solve the problem is: 2 + 3 = 7
    Extracted answer (json format):
        {{
            "equation": "2 + 3 = 7"
        }}
Please extract the answer for the following response:
    Question: {question}
    Model Response: {response}
    You should only output the json without any other texts.
"""

choice_prompt = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
Example1:
    Question: Which answer is right?\n A.1\n B.2\n C.3\n D.4\n Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
    Model Response: The possible answer is: A
    Extracted answer: A
Example2:
    Question: Which answer is right?\n A.1\n B.2\n C.3\n D.4\n Please answer the question and provide all correct option letter, e.g., A, B, C, D, at the end. Find all possible answers.
    Model Response: The possible answer is: A, C
    Extracted answer: [A, C]

Please extract the answer for the following response:
    Question: {question}
    Model Response: {response}
    Extracted answer:
    You should only output the answer without any other texts.
"""

open_question_prompt = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
Example1:
    Question: What value should be? Please answer the question and provide the final value, e.g., 2, 2.10, -3, at the end.
    Model Response: The possible value is -4
    Extracted answer: -4
Example2:
    Question: What value should be? Please answer the question and provide the final value, e.g., 2, 2.10, -3, at the end.
    Model Response: The possible value is $2\sqrt{{3}}$
    Extracted answer: 2\sqrt{{3}}
Example3:
    Question: What value should be? Please answer the question and provide the final value, e.g., 2, 2.10, -3, at the end.
    Model Response: The possible value is 3.28
    Extracted answer: 3.28

Please extract the answer for the following response:
    Question: {question}
    Model Response: {response}
    Extracted answer:
    You should only output the answer without any other texts.
"""

hashi_extraction_prompt = """Extract all bridge connections from the Hashi puzzle solution text and format them as a structured JSON list. Follow these rules:

1. **Input**:
   - `solution`: Text describing bridges between islands using various formats (e.g., "c1 - c3", "a1到g1", "between b2 and b4").

2. **Output Requirements**:
   - Return a JSON list of dictionaries in this format:
     ```JSON
     [{{"start": "a1", "end": "b1", "number": 2}}, ...]
     ```
   - Include ALL bridges explicitly described in `solution`.
   - Use 1-based row numbers and letter-based columns (e.g., "c3" not "3c").
   - Normalize coordinate formats (e.g., "(1,c)" → "c1", "d,4" → "d4").

3. **Parsing Rules**:
   - Capture bridge counts (1 or 2) from phrases like:
     - "two bridges"
     - "1 bridge"
     - "double bridge"
   - Handle directional phrases:
     - "from X to Y"
     - "X connects to Y"
     - "X-Y bridge"

4. **Edge Cases**:
   - The bridge corresponding to the following plot will be skipped directly:
     - No bridges can be parsed
     - Ambiguous connections (unclear start/end)
     - Invalid coordinate formats

**Response Format**:
1. Return ONLY the JSON list.
2. Do not include any additional text, explanations, or formatting beyond the JSON list.

**Examples**:
Solution:
"Draw two bridges from a1 to b1, and one bridge between f6-f8"
Output:
[{{"start": "a1", "end": "b1", "number": 2}}, {{"start": "f6", "end": "f8", "number": 1}}]

Solution:
"Connect c3 with d3 using a double bridge"
Output:
[{{"start": "c3", "end": "d3", "number": 2}}]

Now, extract the answer of the following response:
Solution:
{response}
Output:
"""

skyscraper_extraction_prompt = """Extract the final answer from the given solution. The solution will contain a detailed solution to a Sudoku-like puzzle, including step-by-step explanations and a final filled grid representation. Your task is to identify and extract only the final answer, which is presented as a matrix (list of lists or equivalent) at the end of the solution.

### Requirements:
1. Only return the final filled matrix, formatted as a JSON list of lists.
2. Do not include any additional text, explanations, or formatting beyond the matrix itself.
3. The final answer usually follows phrases such as 'Final Filled Grid' or 'Summary as a Single Filled Matrix'.

### Example 1:
Solution:
To solve the 3x3 Skyscrapers puzzle, we need to place the numbers 1, 2, and 3 in each row and column while adhering to the given clues. Let's break down the solution step by step.\n\n### Step-by-Step Solution:\n\n1. **Analyze the clues:**\n   - Top row: 3, 1, 2\n   - Left column: 2, 2, 1\n   - Right column: 2, 1, 3\n   - Bottom row: 1, 2, 2\n\n2. **Start with the top row:**\n   - The top row has a clue of 3 on the left, meaning all three skyscrapers are visible. This implies the order must be 1, 2, 3 from left to right.\n   - The top row is: [1, 2, 3]\n\n3. **Analyze the left column:**\n   - The left column has a clue of 2 at the top, meaning two skyscrapers are visible. Given the top row is [1, 2, 3], the second row must start with a 3 to block the view of the third row.\n   - The left column is: [1, 3, 2]\n\n4. **Analyze the right column:**\n   - The right column has a clue of 3 at the bottom, meaning all three skyscrapers are visible. This implies the order must be 1, 2, 3 from top to bottom.\n   - The right column is: [3, 2, 1]\n\n5. **Fill in the remaining cells:**\n   - The second row has a clue of 2 on the left, and we already have a 3 in the first column. The remaining cells must be 1 and 2 in some order. Given the right column clue of 2, the second row must be [3, 1, 2].\n   - The third row has a clue of 1 on the left, meaning the first cell must be 2 to block the view of the other cells. The remaining cells must be 3 and 1 in some order. Given the right column clue of 3, the third row must be [2, 3, 1].\n\n### Final Solution:\n\nThe completed 3x3 Skyscrapers puzzle is:\n\n```\n1 2 3\n3 1 2\n2 3 1\n```\n\n### Summary as a single filled matrix:\n\n```python\n[[1, 2, 3], [3, 1, 2], [2, 3, 1]]\n```

Output:
[[1, 2, 3], [3, 1, 2], [2, 3, 1]]

### Example 2:
Solution:
To solve the 4x4 Skyscrapers puzzle, we need to place the numbers 1, 2, 3, and 4 in each row and column while adhering to the given clues. Let's break down the solution step by step.\n\n### Step-by-Step Solution:\n\n1. **Analyze the clues:**\n   - Top row: Clue '2' on the left means we can see 2 skyscrapers from the left.\n   - Bottom row: Clue '4' on the right means we can see 4 skyscrapers from the right.\n   - Left column: Clue '2' on the top means we can see 2 skyscrapers from the top.\n   - Right column: Clue '2' on the bottom means we can see 2 skyscrapers from the bottom.\n\n2. **Start with the bottom row:**\n   - Since we can see 4 skyscrapers from the right, the bottom row must be in ascending order from left to right: [1, 2, 3, 4].\n\n3. **Fill the bottom row:**\n   - Row 4: [1, 2, 3, 4]\n\n4. **Analyze the top row:**\n   - We can see 2 skyscrapers from the left, so the tallest skyscraper (4) must be in the second position from the left to block the view of the third and fourth skyscrapers.\n   - Possible configurations: [3, 4, 1, 2] or [2, 4, 1, 3]\n\n5. **Analyze the left column:**\n   - We can see 2 skyscrapers from the top, so the tallest skyscraper (4) must be in the second position from the top to block the view of the third and fourth skyscrapers.\n   - Possible configurations: [3, 4, 1, 2] or [2, 4, 1, 3]\n\n6. **Combine the clues:**\n   - Top row: [3, 4, 1, 2]\n   - Left column: [3, 4, 1, 2]\n\n7. **Fill the remaining cells:**\n   - Row 1: [3, 4, 1, 2]\n   - Row 2: [4, 1, 2, 3]\n   - Row 3: [2, 3, 4, 1]\n   - Row 4: [1, 2, 3, 4]\n\n### Final Solution:\n\n```\n[[3, 4, 1, 2],\n [4, 1, 2, 3],\n [2, 3, 4, 1],\n [1, 2, 3, 4]]\n```\n\nThis matrix satisfies all the given clues and constraints of the Skyscrapers puzzle.

Output:
[[3, 4, 1, 2], [4, 1, 2, 3], [2, 3, 4, 1], [1, 2, 3, 4]]

Now, please extract the answer for the following solution:
Solution:
{response}

Output:
"""

sudoku_4_extraction_prompt = """Extract the final answer from the given solution. The solution will contain a detailed solution to a Sudoku-like puzzle, including step-by-step explanations and a final filled grid representation. Your task is to identify and extract only the final answer, which is presented as a matrix (list of lists or equivalent) at the end of the solution.

### Requirements:
1. Only return the final filled matrix, formatted as a JSON list of lists.
2. Do not include any additional text, explanations, or formatting beyond the matrix itself.
3. The final answer usually follows phrases such as 'Final Filled Grid' or 'Summary as a Single Filled Matrix'.

### Example 1:
Solution:
To solve the given Jigsaw Sudoku puzzle, we need to ensure that each row, each column, and each 2x2 sub-grid contains the digits 1 through 4 exactly once. Let's fill in the grid step by step.\n\nStep-by-Step Solution:\n1. **Identify the given numbers and their positions:**\n   - Top row: 1, 3\n   - Second row: 3\n   - Third row: (empty)\n   - Fourth row: 2, 3, 1\n\n2. **Fill in the top row:**\n   - The top row already has 1 and 3. The remaining digits are 2 and 4.\n   - The first column already has a 3, so the first cell in the top row cannot be 3. Therefore, it must be 2.\n   - The fourth column already has a 3, so the fourth cell in the top row cannot be 3. Therefore, it must be 4.\n   - Top row: [2, 1, 4, 3]\n\n### Final Filled Matrix:\n[[2, 1, 4, 3], [1, 4, 2, 3], [4, 2, 1, 3], [2, 3, 4, 1]]

Output:
[[2, 1, 4, 3], [1, 4, 2, 3], [4, 2, 1, 3], [2, 3, 4, 1]]

### Example 2:
Solution:
1. **Identify the given numbers and their positions:**\n   - Top row: 1, 3\n   - Second row: 3\n   - Third row: (empty)\n   - Fourth row: 2, 3, 1\n\n2. **Fill in the top row:**\n   - The top row already has 1 and 3. The remaining digits are 2 and 4.\n   - The first column already has a 3, so the first cell in the top row cannot be 3. Therefore, it must be 2.\n   - The fourth column already has a 3, so the fourth cell in the top row cannot be 3. Therefore, it must be 4.\n   - Top row: 2, 1, 4, 3\n\n3. **Fill in the second row:**\n   - The second row already has a 3. The remaining digits are 1, 2, and 4.\n   - The first column already has a 2 and a 3, so the first cell in the second row cannot be 2 or 3. Therefore, it must be 1.\n   - The second column already has a 1, so the second cell in the second row cannot be 1. Therefore, it must be 4.\n   - The fourth column already has a 3 and a 4, so the fourth cell in the second row cannot be 3 or 4. Therefore, it must be 2.\n   - Second row: 1, 4, 2, 3\n\n4. **Fill in the third row:**\n   - The third row is empty. The remaining digits are 1, 2, 3, and 4.\n   - The first column already has a 2 and a 1, so the first cell in the third row cannot be 2 or 1. Therefore, it must be 4.\n   - The second column already has a 1 and a 4, so the second cell in the third row cannot be 1 or 4. Therefore, it must be 2.\n   - The third column already has a 4 and a 2, so the third cell in the third row cannot be 4 or 2. Therefore, it must be 1.\n   - The fourth column already has a 3 and a 2, so the fourth cell in the third row cannot be 3 or 2. Therefore, it must be 3.\n   - Third row: 4, 2, 1, 3\n\n5. **Fill in the fourth row:**\n   - The fourth row already has 2, 3, and 1. The remaining digit is 4.\n   - Fourth row: 2, 3, 4, 1\n\n### Final Filled Matrix:\n```python\n[\n    [2, 1, 4, 3],\n    [1, 4, 2, 3],\n    [4, 2, 1, 3],\n    [2, 3, 4, 1]\n]\n```\n\n### Summary:\nThe completed Jigsaw Sudoku puzzle is:\n```python\n[\n    [2, 1, 4, 3],\n    [1, 4, 2, 3],\n    [4, 2, 1, 3],\n    [2, 3, 4, 1]\n]\n```

Output:
[[2, 1, 4, 3],[1, 4, 2, 3],[4, 2, 1, 3],[2, 3, 4, 1]]

Now, extract the answer of the following response:

Solution:
{response}

Output:
"""

sudoku_6_extraction_prompt = """Extract the final answer from the given solution. The solution will contain a detailed solution to a Sudoku-like puzzle, including step-by-step explanations and a final filled grid representation. Your task is to identify and extract only the final answer, which is presented as a matrix (list of lists or equivalent) at the end of the solution.

### Requirements:
1. Only return the final filled matrix, formatted as a JSON list of lists.
2. Do not include any additional text, explanations, or formatting beyond the matrix itself.
3. The final answer usually follows phrases such as 'Final Filled Grid' or 'Summary as a Single Filled Matrix'.

### Example 1:
Solution:
To solve the given Jigsaw Sudoku puzzle, we need to ensure that each row, each column, and each 2x3 sub-grid contains the digits 1 through 6 exactly once. Let's solve it step by step.\n\n### Step-by-Step Solution:\n\n1. **Identify the sub-grids and initial constraints:**\n   - The grid is divided into 2x3 sub-grids.\n   - We need to fill in the empty cells while satisfying the Sudoku rules.\n\n2. **Fill in the first row:**\n   - The first row has the digits 2 and 4 already placed.\n   - The remaining digits are 1, 3, 5, and 6.\n   - Considering the constraints of the sub-grids and columns, we can place the digits as follows:\n     - First row: [5, 3, 2, 1, 6, 4]\n\n3. **Fill in the second row:**\n   - The second row has the digits 6 and 1 already placed.\n   - The remaining digits are 2, 3, 4, and 5.\n   - Considering the constraints, we can place the digits as follows:\n     - Second row: [4, 6, 1, 5, 3, 2]\n\n4. **Fill in the third row:**\n   - The third row has the digits 1, 3, and 6 already placed.\n   - The remaining digits are 2, 4, and 5.\n   - Considering the constraints, we can place the digits as follows:\n     - Third row: [1, 3, 5, 6, 2, 4]\n\n5. **Fill in the fourth row:**\n   - The fourth row has the digits 6, 3, and 1 already placed.\n   - The remaining digits are 2, 4, and 5.\n   - Considering the constraints, we can place the digits as follows:\n     - Fourth row: [2, 5, 6, 4, 3, 1]\n\n6. **Fill in the fifth row:**\n   - The fifth row has the digits 4 and 2 already placed.\n   - The remaining digits are 1, 3, 5, and 6.\n   - Considering the constraints, we can place the digits as follows:\n     - Fifth row: [3, 1, 4, 2, 5, 6]\n\n7. **Fill in the sixth row:**\n   - The sixth row has the digits 6 and 3 already placed.\n   - The remaining digits are 1, 2, 4, and 5.\n   - Considering the constraints, we can place the digits as follows:\n     - Sixth row: [6, 2, 1, 3, 4, 5]\n\n### Final Filled Matrix:\n\n```python\n[\n    [5, 3, 2, 1, 6, 4],\n    [4, 6, 1, 5, 3, 2],\n    [1, 3, 5, 6, 2, 4],\n    [2, 5, 6, 4, 3, 1],\n    [3, 1, 4, 2, 5, 6],\n    [6, 2, 1, 3, 4, 5]\n]\n```

Output:
[[5, 3, 2, 1, 6, 4],[4, 6, 1, 5, 3, 2],[1, 3, 5, 6, 2, 4],[2, 5, 6, 4, 3, 1],[3, 1, 4, 2, 5, 6],[6, 2, 1, 3, 4, 5]]

### Example 2:
Solution:
To solve this 6×6 Jigsaw Sudoku puzzle, we need to ensure that:\n\n1. Each row contains the digits 1–6 exactly once.\n2. Each column contains the digits 1–6 exactly once.\n3. Each 2×3 sub-grid contains the digits 1–6 exactly once.\n\nHere is the completed solution:\n\n### Completed Sudoku Grid:\n```\n1 3 4 2 5 6\n6 2 5 1 3 4\n5 6 2 3 4 1\n3 4 6 5 1 2\n4 5 1 6 2 3\n2 1 3 4 6 5\n```\n\n### Summary as a Matrix:\n```python\n[\n    [1, 3, 4, 2, 5, 6],\n    [6, 2, 5, 1, 3, 4],\n    [5, 6, 2, 3, 4, 1],\n    [3, 4, 6, 5, 1, 2],\n    [4, 5, 1, 6, 2, 3],\n    [2, 1, 3, 4, 6, 5]\n]\n```

Output:
[[1, 3, 4, 2, 5, 6],[6, 2, 5, 1, 3, 4],[5, 6, 2, 3, 4, 1],[3, 4, 6, 5, 1, 2],[4, 5, 1, 6, 2, 3],[2, 1, 3, 4, 6, 5]]

Now, extract the answer of the following response:

Solution:
{response}

Output:
"""

yinyang_extraction_prompt = """Extract the final answer from the given solution. The solution will contain a detailed solution to a Sudoku-like puzzle, including step-by-step explanations and a final filled grid representation. Your task is to identify and extract only the final answer, which is presented as a matrix (list of lists or equivalent) at the end of the solution.

### Requirements:
1. Only return the final filled matrix, formatted as a JSON list of lists.
2. Do not include any additional text, explanations, or formatting beyond the matrix itself.
3. The final answer usually follows phrases such as 'Final Filled Grid' or 'Summary as a Single Filled Matrix'.

### Example 1:
Solution:
To solve the Yin-Yang puzzle, we need to place black and white stones at the intersections of the grid lines while adhering to the given rules. Let's analyze the current state of the grid and determine the placement of the remaining stones.\n\n### Step-by-Step Solution:\n\n1. **Initial Setup:**\n   - The grid is a 6x6 matrix.\n   - Some cells already have black (1) and white (0) stones.\n\n2. **Rule Analysis:**\n   - All black stones must be orthogonally connected.\n   - All white stones must be orthogonally connected.\n   - No 2x2 region can be monochromatic.\n   - Existing stones cannot be moved.\n\n3. **Placement Strategy:**\n   - Start by ensuring the connectivity of black and white stones.\n   - Avoid creating monochromatic 2x2 regions.\n\n### Detailed Placement:\n\n- **Top-left corner (a1):**\n  - Place a white stone to avoid a monochromatic 2x2 region with the black stone at (b1).\n\n- **Top-right corner (f1):**\n  - Place a white stone to avoid a monochromatic 2x2 region with the black stone at (e1).\n\n- **Middle section:**\n  - Ensure connectivity of black and white stones while avoiding monochromatic 2x2 regions.\n\n### Final Matrix:\n\nAfter placing the stones according to the rules, the final matrix is:\n\n```\n[\n  [0, 1, 0, 1, 0, 0],\n  [0, 0, 1, 1, 0, 0],\n  [1, 0, 1, 0, 1, 0],\n  [0, 1, 0, 1, 0, 1],\n  [0, 0, 1, 0, 1, 0],\n  [1, 0, 0, 1, 0, 0]\n]\n```\n\n### Summary:\n\nThe completed solution for the Yin-Yang puzzle is represented as a single filled matrix:\n\n```python\n[\n  [0, 1, 0, 1, 0, 0],\n  [0, 0, 1, 1, 0, 0],\n  [1, 0, 1, 0, 1, 0],\n  [0, 1, 0, 1, 0, 1],\n  [0, 0, 1, 0, 1, 0],\n  [1, 0, 0, 1, 0, 0]\n]\n```

Output:
[[0, 1, 0, 1, 0, 0],[0, 0, 1, 1, 0, 0],[1, 0, 1, 0, 1, 0],[0, 1, 0, 1, 0, 1],[0, 0, 1, 0, 1, 0],[1, 0, 0, 1, 0, 0]]

### Example 2:
Solution:
To solve the Yin-Yang puzzle, we need to ensure that the rules are followed:\n\n1. All black stones form a single orthogonally connected group.\n2. All white stones form a single orthogonally connected group.\n3. No 2x2 region is monochromatic.\n4. The positions of the pre-placed stones cannot be changed.\n\n### Step-by-Step Solution:\n\n#### Initial Setup:\nThe given grid is a 6x6 grid with some pre-placed black and white stones. We\'ll represent the grid as follows:\n- "1" for black stones.\n- "0" for white stones.\n- Empty cells will be filled as we solve the puzzle.\n\n#### Pre-placed Stones:\nFrom the diagram:\n- Black stones ("1") are at: (a1, b1, c1, c2, d2, d3, e3).\n- White stones ("0") are at: (a3, a4, a5, b4, b5, c4, d4, e4).\n\n#### Solving the Puzzle:\nWe will fill the remaining cells while ensuring the rules are satisfied.\n\n---\n\n### Final Solution:\nAfter solving, the completed grid is as follows:\n\n```\n1 1 1 0 0 0\n0 1 0 1 0 0\n0 1 0 1 1 0\n0 0 0 1 0 1\n0 0 0 0 0 1\n1 0 0 0 1 1\n```\n\n### Summary:\nThe solution as a 6x6 matrix is:\n\n```python\n[\n    [1, 1, 1, 0, 0, 0],\n    [0, 1, 0, 1, 0, 0],\n    [0, 1, 0, 1, 1, 0],\n    [0, 0, 0, 1, 0, 1],\n    [0, 0, 0, 0, 0, 1],\n    [1, 0, 0, 0, 1, 1]\n]\n```

Output:
[[1, 1, 1, 0, 0, 0],[0, 1, 0, 1, 0, 0],[0, 1, 0, 1, 1, 0],[0, 0, 0, 1, 0, 1],[0, 0, 0, 0, 0, 1],[1, 0, 0, 0, 1, 1]]

Now, extract the answer of the following response:
Solution:
{response}

Output:
"""

points24_prompt = """
Please read the following examples. Then extract the final formula from the answer to the 24-point game, and type it at the end of the prompt. In the cards, K stands for 13, Q stands for 12, J stands for 11, and A stands for 1. Note you need to use * to represent multiplication sign, / to represent division sign.
Example1:
    Question: Given four playing cards (A, 8, 9, K), each with a value, use any combination of addition, subtraction, multiplication, and division to make the number 24. You must use each card exactly once. Give the final answer as a formula.
    Model Response: The possible answer is (K - 9 - A) × 8 = 24
    Extracted answer: (13-9-1)*8=24
Example2:
    Question: Given four playing cards (3, 8, 9, A), each with a value, use any combination of addition, subtraction, multiplication, and division to make the number 24. You must use each card exactly once. Give the final answer as a formula.
    Model Response: The possible answer is 9 \\div 3 \\times A \\times 8 = 24
    Extracted answer: 9/3*1*8=24

Please extract the final formula from for the following response:
    Question: {question}
    Model Response: {response}
    Extracted answer:
    You should only output the final formula from without any other texts.
"""

mme_reasoning_eval_prompts = {
    "id_answer_pair_prompt": id_answer_pair_prompt,
    "id_answer_pair_prompt_reverse": id_answer_pair_prompt_reverse,
    "coordinate_answer_prompt": coordinate_answer_prompt,
    "multiple_answer_prompt": multiple_answer_prompt,
    "formula_extraction_prompt": formula_extraction_prompt,
    "choice_prompt": choice_prompt,
    "open_question_prompt": open_question_prompt,
    "hashi_extraction_prompt": hashi_extraction_prompt,
    "skyscraper_extraction_prompt": skyscraper_extraction_prompt,
    "sudoku_4_extraction_prompt": sudoku_4_extraction_prompt,
    "sudoku_6_extraction_prompt": sudoku_6_extraction_prompt,
    "yinyang_extraction_prompt": yinyang_extraction_prompt,
    "points24_prompt": points24_prompt
}

def calculate_answer_function_1(response_dict, answer_dict=None):
    try:
        expect_keys = ['A', 'B', 'C']
        for k in expect_keys:
            if k not in response_dict.keys():
                return False
        if response_dict["A"] / response_dict['B'] == 3 and response_dict["C"] / response_dict['B'] == 2:
            return True
        else:
            return False
    except:
        return False

def calculate_answer_function_2(response_dict, answer_dict=None):
    try:
        expect_keys = ['A', 'B', 'C', 'D', 'E', 'F']
        for k in expect_keys:
            if k not in response_dict.keys():
                return False
        if response_dict['A'] + response_dict['B'] + response_dict['C'] != 25:
            return False
        if response_dict['C'] + response_dict['E'] + response_dict['F'] != 25:
            return False
        if response_dict['A'] + response_dict['D'] + response_dict['F'] != 25:
            return False
        return True
    except:
        return False

def calculate_answer_function_3(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = set(response_dict.values())
    expect_values = set(range(1, 16))
    if values != expect_values:
        return False
    diamond_list = [
        ['A', 'D', 'E', 'H'],
        ['C', 'E', 'G', 'H'],
        ['I', 'G', 'J', 'H'],
        ['N', 'M', 'J', 'H'],
        ['O', 'M', 'H', 'L'],
        ['K', 'L', 'H', 'F'],
        ['B', 'F', 'H', 'D']
    ]
    for diamond in diamond_list:
        total_value = 0
        for key in diamond:
            total_value += response_dict[key]
        if total_value != 30:
            return False
    return True

def calculate_answer_function_4(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = set(response_dict.values())
    expect_values = set(range(1, 10))
    if values != expect_values:
        return False
    line_list = [
        ['A', 'B', 'E', 'D'],
        ['B', 'C', 'E', 'F'],
        ['D', 'E', 'G', 'H'],
        ['E', 'F', 'H', 'I'],
        ['B', 'D', 'H', 'F'],
        ['A', 'C', 'I', 'G']
    ]
    for line in line_list:
        total_value = 0
        for key in line:
            total_value += response_dict[key]
        if total_value != 20:
            return False
    return True

def calculate_answer_function_5(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = list(response_dict.values())
    for v in values:
        if v not in [20, 15, 10, 5]:
            return False
    num_dict = {}
    for v in values:
        num_dict[v] = num_dict.get(v, 0) + 1
    if num_dict.get(20, 0) > 5 or num_dict.get(15, 0) > 3 or num_dict.get(10, 0) > 3 or num_dict.get(5, 0) > 6:
        return False

    line_list = [
        ['A', 'B', 'C'],
        ['A', 'G', 'O'],
        ['O', 'P', 'Q'],
        ['C', 'K', 'Q'],
        ['A', 'D', 'I', 'N', 'Q'],
        ['C', 'F', 'I', 'L', 'O'],
        ['B', 'E', 'I', 'M', 'P'],
        ['G', 'H', 'I', 'J', 'K'],

    ]
    for line in line_list:
        total_value = 0
        for key in line:
            total_value += response_dict[key]
        if total_value != 55:
            return False
    return True

def calculate_answer_function_6(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = set(response_dict.values())
    expect_values = set(range(1, 20))
    if values != expect_values:
        return False
    line_list = [
        ['A', 'J', 'S'],
        ['C', 'J', 'Q'],
        ['E', 'J', 'O'],
        ['G', 'J', 'M'],
        ['I', 'J', 'K'],
        ['L', 'J', 'H'],
        ['N', 'J', 'F'],
        ['P', 'J', 'D'],
        ['R', 'J', 'B']
    ]
    for line in line_list:
        total_value = 0
        for key in line:
            total_value += response_dict[key]
        if total_value != 30:
            return False
    return True

def calculate_answer_function_7(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = set(response_dict.values())
    expect_values = set(range(1, 13))
    if values != expect_values:
        return False
    line_list = [
        ['A', 'C', 'F', 'H'],
        ['A', 'D', 'G', 'K'],
        ['B', 'C', 'D', 'E'],
        ['H', 'I', 'J', 'K'],
        ['B', 'F', 'I', 'L'],
        ['E', 'G', 'J', 'L']
    ]
    for line in line_list:
        total_value = 0
        for key in line:
            total_value += response_dict[key]
        if total_value != 26:
            return False
    return True

def calculate_answer_function_8(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = list(response_dict.values())
    for v in values:
        if v < 1 or v > 15:
            return False
    if len(values) != len(set(values)):
        return False
    line_list = [
        ['A', 'B', 'C'],
        ['A', 'F', 'K'],
        ['B', 'D', 'E'],
        ['B', 'E', 'H'],
        ['C', 'H', 'M'],
        ['D', 'G', 'J'],
        ['E', 'G', 'I'],
        ['F', 'I', 'L'],
        ['H', 'J', 'I'],
        ['K', 'L', 'M']
    ]
    for line in line_list:
        total_value = 0
        for key in line:
            total_value += response_dict[key]
        if total_value != 20:
            return False
    return True

def calculate_answer_function_9(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = set(response_dict.values())
    expect_values = set(range(1, 20))
    if values != expect_values:
        return False

    line_list = [
        ['A', 'B', 'C'],
        ['A', 'D', 'H'],
        ['A', 'E', 'J'],
        ['C', 'F', 'J'],
        ['C', 'G', 'L'],
        ['H', 'I', 'J'],
        ['H', 'M', 'Q'],
        ['J', 'K', 'L'],
        ['J', 'N', 'Q'],
        ['J', 'O', 'S'],
        ['L', 'P', 'S'],
        ['Q', 'R', 'S']
    ]
    for line in line_list:
        total_value = 0
        for key in line:
            total_value += response_dict[key]
        if total_value != 22:
            return False
    return True

def calculate_answer_function_10(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = set(response_dict.values())
    expect_values = set(range(1, 7))
    if values != expect_values:
        return False
    line_1 = response_dict['A'] + response_dict['B'] + response_dict['D']
    line_2 = response_dict['D'] + response_dict['E'] + response_dict['F']
    line_3 = response_dict['A'] + response_dict['C'] + response_dict['F']
    return line_1 == line_2 == line_3

def calculate_answer_function_11(response_dict, answer_dict=None):
    expect_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    values = set(response_dict.values())
    expect_values = set([1, 2, 3, 5, 7, 8])
    if values != expect_values:
        return False

    if response_dict['A'] + response_dict['C'] + 9 != 19:
        return False
    if response_dict['B'] + response_dict['C'] + response_dict['D'] != 19:
        return False
    if response_dict['F'] + response_dict['E'] + response_dict['D'] != 19:
        return False
    if response_dict['E'] + response_dict['G'] + 6 != 19:
        return False
    return True

def calculate_answer_function_12(response_dict, answer_dict=None):
    expect_keys = ['A', 'C', 'T', 'D', 'O', 'G', 'P', 'E', 'T', 'S']
    for k in expect_keys:
        if k not in response_dict.keys():
            return False
    num_1 = response_dict['C'] * 100 + response_dict['A'] * 10 + response_dict['T']
    num_2 = response_dict['D'] * 100 + response_dict['O'] * 10 + response_dict['G']
    num_3 = response_dict['P'] * 1000 + response_dict['E'] * 100 + response_dict['T'] * 10 + response_dict['S']
    return num_1 + num_2 == num_3

def match_answer_function(response_dict, answer_dict):
    for key, value in answer_dict.items():
        cur_response = response_dict.get(key, None)
        if isinstance(cur_response, list):
            cur_response = set(cur_response)
        if isinstance(value, list):
            value = set(value)
        if cur_response != value:
            return False
    return True

def match_coordinate_function(response_list, answer_list):
    response_coor = []
    for coor in response_list:
        response_coor.append(str(coor['row']) + str(coor['column']))

    answer_coor = []
    if isinstance(answer_list, dict):
        answer_list = [answer_list]
    for coor in answer_list:
        answer_coor.append(str(coor['row']) + str(coor['column']))

    return response_coor == answer_coor

# answer justify: answer in xxx
def multiple_match_function(response_dict, answer_list):
    assert len(response_dict) == 1
    response = list(response_dict.values())[0]
    if isinstance(response, list):
        response = set(response)
        answer_list = [set(ans) for ans in answer_list]
        return response in answer_list
    else:
        return response in answer_list

def compare_expression_function(response_dict, anwser_list):
    extract_equation = response_dict['equation']
    extract_expr = sympy.parse_expr(extract_equation.split('=')[0], evaluate=False)
    is_same = False
    for answer_eq in anwser_list:
        if is_same == True:
            return True
        is_cur_same = False
        answer_expr = sympy.parse_expr(answer_eq.split('=')[0], evaluate=False)
        if extract_expr.func != answer_expr.func:
            is_cur_same = False
            continue

        # for addition and multiplication, check if the set of arguments is the same
        if extract_expr.func in (sympy.Add, sympy.Mul):
            args1 = extract_expr.args
            args2 = answer_expr.args

            # if the number of arguments is different, the expressions are not the same
            if len(args1) != len(args2):
                is_cur_same = False
                continue
            # check if the set of arguments is the same
            is_cur_same = (set(str(arg) for arg in args1) == set(str(arg) for arg in args2))
        # for other functions, check if the expression is the same
        else:
            is_cur_same = (extract_expr == answer_expr)
        is_same = is_cur_same
    return is_same

def choice_function(response, answer):
    try:
        answer = eval(answer)
    except:
        pass
    if isinstance(answer, list):
        answer = ','.join(answer)
    else:
        answer = str(answer)
    response_list = [ans.strip() for ans in response.split(',')]
    answer_list = [ans.strip() for ans in answer.split(',')]
    return set(response_list) == set(answer_list)


def calculate_answer_function_hashi(response_list, answer=None, special_info=None):
    if response_list is None or special_info is None or len(response_list) == 0:
        return False
    bridges = response_list
    # special_info = json.loads(special_info)
    init_islands = special_info["init_islands"]
    # init_islands be like:
    # [
    # {"cord": "b1", "requirement": 2},
    # {"cord": "f8", "requirement": 1}
    # ]

    def is_valid_hashi(init_islands, bridges):
        from collections import defaultdict

        # Helper function to parse coordinates
        def parse_coordinate(cord):
            letter, number = cord[0], int(cord[1:])
            return ord(letter) - ord('a'), number

        # Rule 1: Distinct islands check
        for bridge in bridges:
            if bridge['start'] == bridge['end']:
                return False, "Rule 1 violated: Bridge starts and ends at the same island."

        # Group islands by coordinates for quick lookup
        island_requirements = {island['cord']: island['requirement'] for island in init_islands}
        existing_islands = set(island_requirements.keys())

        # Check for total bridge connections per island
        bridge_count = defaultdict(int)
        connection_count = defaultdict(int)

        # Check each bridge for validity
        for bridge in bridges:
            start = bridge['start']
            end = bridge['end']
            number = bridge['number']

            # Rule 6: Island existence check
            if start not in existing_islands or end not in existing_islands:
                return False, f"Rule 6 violated: One or both islands ({start}, {end}) do not exist."

            # Rule 3: Orthogonal Check
            start_x, start_y = parse_coordinate(start)
            end_x, end_y = parse_coordinate(end)
            if start_x != end_x and start_y != end_y:
                return False, f"Rule 3 violated: Bridge between {start} and {end} is not orthogonal."

            # Rule 4: Maximum two bridges between pair of islands
            pair = tuple(sorted([start, end]))
            connection_count[pair] += number
            if connection_count[pair] > 2:
                return False, f"Rule 4 violated: More than two bridges between {start} and {end}."

            # Count connections for each island
            bridge_count[start] += number
            bridge_count[end] += number

        # Rule 5: Check if each island's requirement is met
        for island, requirement in island_requirements.items():
            if bridge_count[island] != requirement:
                return False, f"Rule 5 violated: Island {island} has {bridge_count[island]} bridges, requires {requirement}."

        # Rule 2: Cross-check - Ensure no crossing bridges
        occupied_lines = set()
        for bridge in bridges:
            start_x, start_y = parse_coordinate(bridge['start'])
            end_x, end_y = parse_coordinate(bridge['end'])

            # Generate the line path between start and end
            if start_x == end_x:  # Vertical line
                y_min, y_max = sorted([start_y, end_y])
                for y in range(y_min + 1, y_max):
                    line_segment = (start_x, y)
                    if line_segment in occupied_lines:
                        return False, f"Rule 2 violated: Crossing bridge at ({chr(start_x + ord('a'))}{y})."
                    occupied_lines.add(line_segment)

            elif start_y == end_y:  # Horizontal line
                x_min, x_max = sorted([start_x, end_x])
                for x in range(x_min + 1, x_max):
                    line_segment = (x, start_y)
                    if line_segment in occupied_lines:
                        return False, f"Rule 2 violated: Crossing bridge at ({chr(x + ord('a'))}{start_y})."
                    occupied_lines.add(line_segment)

        return True, "Hashi: All rules satisfied."

    validity, message = is_valid_hashi(init_islands, bridges)
    print(message)
    return validity


def calculate_answer_function_skyscraper(response_list, answer=None, special_info=None):
    if response_list is None or special_info is None or len(response_list) == 0:
        return False

    grid = response_list
    visible_skyscrapers = special_info["visible_skyscrapers"]
    # visible_skyscrapers be like:
    # [
    #     [3,2,1],
    #     [3,2,1],
    #     [1,None,2],
    #     [1,2,None]
    # ]
    # represent the following case:
    # [
    #     [None,  3,      2,      1,      None],
    #     [3,     None,   None,   None,   1],
    #     [2,     None,   None,   None,   None],
    #     [1,     None,   None,   None,   2],
    #     [None,  1,      2,      None,   None]
    # ]


    # Verify if the base matrix has been changed
    base_matrix = special_info['base_matrix']
    size = len(base_matrix)
    for i in range(size):
        for j in range(size):
            if base_matrix[i][j] is not None and base_matrix[i][j]!=response_list[i][j]:
                # print("Mismatch with base matrix")
                return False


    for i in range(size):
        if len(set(grid[i])) != size or len(set(row[i] for row in grid)) != size:
            # print("repeat height on one row or colomn")
            return False

    def count_higher_than_previous(lst, target_count):
        count = 1  # Start with 1 since the first number is always higher than "nothing before it"
        max_so_far = lst[0]
        for i in range(1, len(lst)):
            if lst[i] > max_so_far:
                count += 1
                max_so_far = lst[i]
        return count == target_count

    # Verify the skyscraper rule
    hori = grid
    vert = [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]

    for index, h in enumerate(visible_skyscrapers[0]):
        if h is None:
            continue
        if not count_higher_than_previous(vert[index], h):
            # print("wrong visible towers")
            return False

    for index, h in enumerate(visible_skyscrapers[1]):
        if h is None:
            continue
        if not count_higher_than_previous(hori[index], h):
            # print("wrong visible towers")
            return False

    for index, h in enumerate(visible_skyscrapers[2]):
        if h is None:
            continue
        if not count_higher_than_previous(hori[index][::-1], h):
            # print("wrong visible towers")
            return False

    for index, h in enumerate(visible_skyscrapers[3]):
        if h is None:
            continue
        if not count_higher_than_previous(vert[index][::-1], h):
            # print("wrong visible towers")
            return False
    # print("Skyscraper: All rules satisfied.")
    return True


def calculate_answer_function_sudoku_4(response_list, answer=None, special_info=None):
    if response_list is None or special_info is None or len(response_list) == 0:
        return False

    def is_valid_sudoku_4x4(grid):
        # Check rows and columns
        for i in range(4):
            if len(set(grid[i])) != 4 or len(set(row[i] for row in grid)) != 4:
                # print("repeat numbers in row or colomn")
                return False

        # Check 2x2 sub-grids
        sub_grids = [
            [grid[r][c] for r in range(2) for c in range(2)],  # Top-left
            [grid[r][c] for r in range(2) for c in range(2, 4)],  # Top-right
            [grid[r][c] for r in range(2, 4) for c in range(2)],  # Bottom-left
            [grid[r][c] for r in range(2, 4) for c in range(2, 4)],  # Bottom-right
        ]
        for sub_grid in sub_grids:
            if len(set(sub_grid)) != 4:
                # print("repeat numbers in sub matrix")
                return False
        # print("Sudoku_4: All rules satisfied.")
        return True

    try:
        # Parse the response string into a Python list
        grid = response_list
        if not isinstance(grid, list) or len(grid) != 4 or not all(len(row) == 4 for row in grid):
            return False

        # Verify if the base matrix has been changed
        base_matrix = special_info['base_matrix']
        for i in range(4):
            for j in range(4):
                if base_matrix[i][j] is not None and base_matrix[i][j]!=grid[i][j]:
                    # print("Mismatch with base matrix")
                    return False

        # Validate the Sudoku grid
        return is_valid_sudoku_4x4(grid)
    except (ValueError, SyntaxError):
        return False


def calculate_answer_function_sudoku_6(response_list, answer=None, special_info=None):
    if response_list is None or special_info is None or len(response_list) == 0:
        return False

    def is_valid_sudoku_6x6(grid):
        # Check rows and columns
        for i in range(6):
            if len(set(grid[i])) != 6 or len(set(row[i] for row in grid)) != 6:
                return False

        # Check 2x2 sub-grids
        sub_grids = [
            [grid[r][c] for r in range(2) for c in range(3)],  # Top-left
            [grid[r][c] for r in range(2) for c in range(3, 6)],  # Top-right
            [grid[r][c] for r in range(2, 4) for c in range(3)],  # Middle-left
            [grid[r][c] for r in range(2, 4) for c in range(3, 6)],  # Middle-right
            [grid[r][c] for r in range(4, 6) for c in range(3)],  # Bottom-left
            [grid[r][c] for r in range(4, 6) for c in range(3, 6)],  # Bottom-right
        ]
        for sub_grid in sub_grids:
            if len(set(sub_grid)) != 6:
                return False
        return True

    try:
        # Parse the response string into a Python list
        grid = response_list
        if not isinstance(grid, list) or len(grid) != 6 or not all(len(row) == 6 for row in grid):
            return False

        # Verify if the base matrix has been changed
        base_matrix = special_info['base_matrix']
        for i in range(6):
            for j in range(6):
                if base_matrix[i][j] is not None and base_matrix[i][j]!=grid[i][j]:
                    return False

        # Validate the Sudoku grid
        return is_valid_sudoku_6x6(grid)
    except (ValueError, SyntaxError):
        return False


def calculate_answer_function_yinyang(response_list, answer=None, special_info=None):
    if response_list is None or special_info is None or len(response_list) == 0:
        return False

    def is_connected(matrix, value):
        rows, cols = len(matrix), len(matrix[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or matrix[r][c] != value:
                return
            visited[r][c] = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs(r + dr, c + dc)

        # Find the first cell with the given value
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == value:
                    dfs(r, c)
                    break
            else:
                continue
            break

        # Check if all cells with the given value are visited
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == value and not visited[r][c]:
                    return False
        return True

    def has_2x2_block(matrix, value):
        rows, cols = len(matrix), len(matrix[0])
        for r in range(rows - 1):
            for c in range(cols - 1):
                # Check if a 2x2 block is formed by the same value
                if (
                    matrix[r][c] == value and
                    matrix[r][c + 1] == value and
                    matrix[r + 1][c] == value and
                    matrix[r + 1][c + 1] == value
                ):
                    return True
        return False

    # Check if the response is a 6x6 matrix
    if not isinstance(response_list, list) or len(response_list) != 6 or any(len(row) != 6 for row in response_list):
        return False

    # Verify if the base matrix has been changed
    base_matrix = special_info['base_matrix']
    for i in range(6):
        for j in range(6):
            if base_matrix[i][j] is not None and base_matrix[i][j]!=response_list[i][j]:
                print("Mismatch with base matrix")
                return False

    # Check if 0s and 1s form connected components
    if not is_connected(response_list, 0) or not is_connected(response_list, 1):
        print("not connected")
        return False

    # Check if there are any 2x2 blocks of 0s or 1s
    if has_2x2_block(response_list, 0) or has_2x2_block(response_list, 1):
        print("exist 2x2 block")
        return False
    print("Yinyang: All rules satisfied")
    return True

def judge_24points_function(response, answer=None, special_info=None):
    left_formula = response.replace(' ', '').split('=')[0]
    try:
        if eval(left_formula) == 24:
            numbers = re.findall(r'\d+', left_formula)
            numbers = list(map(int, numbers))
            numbers.sort()
            expect_numbers = special_info['init_cards']
            expect_numbers.sort()
            if numbers == expect_numbers:
                return True
            else:
                return False
        else:
            return False

    except:
        return False

mme_reasoning_eval_functions = {
    "calculate_answer_function_1": calculate_answer_function_1,
    "calculate_answer_function_2": calculate_answer_function_2,
    "calculate_answer_function_3": calculate_answer_function_3,
    "calculate_answer_function_4": calculate_answer_function_4,
    "calculate_answer_function_5": calculate_answer_function_5,
    "calculate_answer_function_6": calculate_answer_function_6,
    "calculate_answer_function_7": calculate_answer_function_7,
    "calculate_answer_function_8": calculate_answer_function_8,
    "calculate_answer_function_9": calculate_answer_function_9,
    "calculate_answer_function_10": calculate_answer_function_10,
    "calculate_answer_function_11": calculate_answer_function_11,
    "calculate_answer_function_12": calculate_answer_function_12,
    "match_answer_function": match_answer_function,
    "multiple_match_function": multiple_match_function,
    "compare_expression_function": compare_expression_function,
    "match_coordinate_function": match_coordinate_function,
    "choice_function": choice_function,
    "calculate_answer_function_hashi": calculate_answer_function_hashi,
    "calculate_answer_function_skyscraper": calculate_answer_function_skyscraper,
    "calculate_answer_function_sudoku_4": calculate_answer_function_sudoku_4,
    "calculate_answer_function_sudoku_6": calculate_answer_function_sudoku_6,
    "calculate_answer_function_yinyang": calculate_answer_function_yinyang,
    "judge_24points_function": judge_24points_function
}


def extract_json_from_response(text):
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'```JSON\s*(.*?)\s*```'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json_str
    return None


def MMEReasoning_extract(model, line):
    res = None
    question, response = line['question'], line['prediction']
    eval_function, eval_prompt = line['function_id'], line['prompt_id']
    if eval_function == None:
        if line['question_type'].lower() == 'choice':
            eval_prompt, eval_function = 'choice_prompt', 'choice_function'
        elif line['question_type'].lower() == 'open':
            eval_prompt, eval_function = 'open_question_prompt', 'open_function'
        else:
            raise NotImplementedError(f"Question type that requires specifying the function_id: {line['question_type']}")

    prompt = mme_reasoning_eval_prompts[eval_prompt].format(question=question, response=response)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        if prediction == FAIL_MSG:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
            return dict(log=log, res='')
        res = model.generate(prompt, temperature=i * 0.5)
        if eval_prompt not in ['open_question_prompt', 'choice_prompt', 'points24_prompt']:
            try:
                json.loads(res)
            except:
                try:
                    json.loads(extract_json_from_response(res))
                    res = extract_json_from_response(res)
                except:
                    continue

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')

def MMEReasoning_openeval(model, line):
    prompt_openeval = """
Please read the following example. Then judge the answer and type it at the end of the prompt.
Below are two examples. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note:
    Different expressions of the same number should also be considered consistent, for example, \\frac{{7}}{{2}} and 3.5.
    If a conversion results in a decimal approximation, the expressions can be considered consistent if the values are equal up to two decimal places, for example, \\sqrt{{3}} and 1.73.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.\n
Example 1:
    [Question]: What is the minimize length of the line?
    [Standard answer]: \\sqrt{{2}}
    [Model answer]: 1.414
    [Judgement]: 1
Example 2:
    [Question]: Given an image of a 3x3 maze. How to reach the end cell marked 'E' from the start cell is marked 'S'.
    [Standard answer]: ['Left', 'Right']
    [Model answer]: 'Left', 'Right'
    [Judgement]: 1

Now, judge the anwser for the following question:
    [Question]: {question}
    [Standard answer]: {answer}
    [Model answer]: {response}
    [Judgement]:
You should only output the judgement without any other texts.
"""
    log = ''
    retry = 5
    prompt = prompt_openeval.format(question=line['question'], answer=line['answer'], response=line['res'])
    for i in range(retry):
        try:
            prediction = line['res']
            if prediction == None or FAIL_MSG in prediction:
                log += f'Try {i}: output is {prediction}, failed to parse.\n'
                return dict(log_score=log, score=False)
            res = model.generate(prompt, temperature=i * 0.5)
            if FAIL_MSG in res or res.strip() not in ['0', '1']:
                log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
            else:
                log += 'Succeed'
                return dict(log_score=log, score=int(res) == 1)
        except:
            continue
    log += 'All 5 retries failed.\n'
    return dict(log_score=log, score=False)


def MMEReasoning_acc(result_file):
    df = load(result_file)

    capabilities = ['planning and exploring', 'calculation', 'spatial-temporal', 'casual chaining analysis', 'pattern analysis']
    reasoning_types = ['inductive', 'deductive', 'abductive']

    res = defaultdict(list)

    # Overall Acc
    res['Overall'].append(np.mean(df['score']) * 100)
    for capability in capabilities:
        sub = df[df['capability'].apply(lambda x: capability in x)]
        res[capability].append(np.mean(sub['score']) * 100)

    for r_type in reasoning_types:
        sub = df[df['reasoning_type'].apply(lambda x: r_type in x)]
        res[r_type].append(np.mean(sub['score']) * 100)

    return pd.DataFrame(res)
