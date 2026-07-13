PROMPT_SOKOBAN_IMAGE = """
Your task is to solve the Sokoban puzzle according to the rules and current state shown in the image:

Game Rules:
1. You are the player and can move up, down, left, or right
2. You can push boxes but only one at a time
3. You cannot pull boxes
4. Boxes can only be pushed if there's an empty space behind them
5. The goal is to push all boxes onto target positions
8. Walls cannot be moved through or pushed

You will be given an image, in the image:

1. Red squares represent the player
2. Pink squares represent boxes
3. Green squares represent target positions
4. White squares represent empty spaces that can be moved into
4. Gray blocks represent walls

Direction Definitions:
- "up": Move up
- "down": Move down
- "left": Move left
- "right": Move right

Current Sokoban State can be seen in the image shown below:

Output Format Requirements:
1. Your final answer should be in the format of a space-separated sequence of moves like:
up right down left
2. You should put your thinking process in `<think></think>` tags,
and the final answer in `<answer></answer>` tags.

Now, think step by step, and provide your solution in the format shown above.
"""


PROMPT_MAZE_IMAGE = """
Your task is to solve the maze game according to the rules and current state below:

Game Rules:
1. The maze consists of a grid of cells
2. Walls are represented by **bold black line** between cells, not as cells themselves
3. You can move horizontally or vertically between adjacent cells if there is no wall between them
4. You can only move through one cell at a time in any direction
5. The goal is to find a path from the start cell (S) to the end cell (E)

Direction Definitions:
- "up": Move to the cell above the current position (toward the top of the maze, decreasing row number)
- "down": Move to the cell below the current position (toward the bottom of the maze, increasing row number)
- "left": Move to the cell to the left of the current position (decreasing column number)
- "right": Move to the cell to the right of the current position (increasing column number)

Current Maze State:
The maze is represented in the image shown below

In this representation:
- green circule marks the start position
- red cross marks the end position

Output Format Requirements:
1. Your final answer should be in the format like: right down left up
2. You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.

Now, think step by step, and provide your solution in the format shown above.
"""

PROMPT_15PUZZLE_IMAGE = """Your task is to solve the 15-puzzle game according to the rules and current state below:

Let me explain the 15-puzzle game rules and the current puzzle state:

Game Rules:
1. The puzzle is played on a 4x4 grid with 15 numbered tiles and one empty space
2. You can only move tiles horizontally or vertically into the empty space
3. The goal is to arrange the tiles in numerical order with:
   - First row: 1, 2, 3, 4
   - Second row: 5, 6, 7, 8
   - Third row: 9, 10, 11, 12
   - Fourth row: 13, 14, 15, empty space

Coordinate System:
- The grid positions are numbered from left to right and top to bottom
- Columns (horizontal): numbered 1, 2, 3, 4 from left to right
- Rows (vertical): numbered 1, 2, 3, 4 from top to bottom
- Each position can be identified by its row and column (row, column)

Current Puzzle State:
The initial_state is represented in the image shown below

Output Format Requirements:
"up" means the tile below the empty space moves up into the empty space
"down" means the tile above the empty space moves down into the empty space
"left" means the tile to the right of the empty space moves left into the empty space
"right" means the tile to the left of the empty space moves right into the empty space

Your final answer format should be given like: up down up left right ,etc.
You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.

Now, think step by step, and provide your solution in the format shown above.
"""

PROMPT_HANOI_IMAGE = """Your task is to solve the hanoi game according to the rules and current state below:

Let me explain the Tower of Hanoi puzzle rules and the current state:

Game Rules:
1. The Tower of Hanoi consists of three pegs (numbered 1, 2, and 3) and n(maybe 3 or 4 or 5) disks of different sizes
(from 1 to n)
2. Disks are stacked on pegs with larger disks always below smaller ones
3. Only one disk can be moved at a time, from the top of one peg to the top of another
4. A larger disk cannot be placed on top of a smaller disk

Current Hanoi State:
The current state of the Tower of Hanoi is in the image shown below

Goal State:
## For 3 disks

[
   [],
   [],
   [3, 2, 1],
]

## For 4 disks
[
   [],
   [],
   [4, 3, 2, 1],
]

## For 5 disks
[
   [],
   [],
   [5, 4, 3, 2, 1],
]


In this text representation:
- Each array [] represents a peg (from 1 to 3)
- Numbers inside the arrays represent disks (higher numbers = larger disks)
- The first/top elements in an array are at the bottom of the peg
- The last/bottom elements in an array are at the top of the peg

Output Format Requirements:
1. Your final solution format should be given like:(x,y) (x,y) (x,y)...,
where x is the disk number and y is the destination peg number
2. You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.

Now, think step by step, and provide your solution in the format shown above.

"""

PROMPT_WORDSEARCH_IMAGE = """Your task is to solve the wordsearch game according to the rules and current state below:


## Task
You are given a word search puzzle.
Your task is to find all the listed words hidden in the grid and provide their exact locations in the specified format.

## Rules of WordDescription Search
1. Words can be hidden horizontally, vertically, or diagonally.
2. Words can read forwards or backwards.
3. Words always follow a straight line (no zigzagging).
4. Each word's location should be identified by:
   - The starting position (coordinate where the first letter appears)
   - The direction in which the word extends

## Coordinate System
- The grid uses coordinates where (x, y) represents the position.
- x-axis: Numbers from 1 to width, running horizontally from left to right.
- y-axis: Numbers from 1 to height, running vertically from top to bottom.
- Example: Position (3, 4) means column 3 from left, row 4 from top.


## Direction Notation
- N: North (upward)
- S: South (downward)
- E: East (rightward)
- W: West (leftward)
- NE: Northeast (up and right)
- NW: Northwest (up and left)
- SE: Southeast (down and right)
- SW: Southwest (down and left)


WordSearch State:
The current state of the WordSearch is shown in the image given below


Output Format Requirements:
1. Your final answer format should be given like: WORD DIRECTION @ (x, y),
where WORD is the word you found,
DIRECTION is the direction in which the word extends,
and (x, y) is the starting position of the word.
2. You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.

Now, think step by step, and provide your solution in the format shown above.
"""


PROMPT_NUMBRIX_IMAGE = """
Your task is to solve the Numbrix puzzle based on the following rules and the current state:

### Game Rules:
1. Numbrix is played on a square grid, where some cells are already filled with numbers.
2. You must fill in the empty cells with numbers to create a continuous path from 1 to the highest number
(grid size squared).
3. The numbers must be adjacent either horizontally or vertically (not diagonally).
4. Each number can only be used once.
5. The path must form a single continuous sequence where consecutive numbers are adjacent.
6. **Not every empty cell needs to be filled.** In some cases, leaving some cells empty may be required,
depending on the puzzle configuration.

### Current Numbrix State:
The current state of the Numbrix puzzle is shown in the image below.

In this representation:
- Filled cells contain the given numbers.
- Empty cells are blank spaces.
- Your goal is to fill the empty cells according to the rules, but remember,
**not every empty cell needs to be filled**.

### Output Format Requirements:
3. The final answer should be the completed grid with all numbers correctly filled in,
maintaining a clear grid format with numbers aligned in rows and columns.
4. **Do not add extra spaces inside the grid cells.** For example, ensure that `|3|` remains `|3|`, not `| 3 |`.

### Example answer format for a 5x5 grid:
|11|10|9|2|3|
|12|13|8|1|4|
|15|14|7|6|5|
|16|19|20|23|24|
|17|18|21|22|25|

You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.

Now, think step by step, and provide your solution in the format shown above.

"""


PROMPT_MINESWEEPER_IMAGE = """

Your task is to solve the Minesweeper puzzle according to the rules and the current state below:

**Game Rules:**
1. Minesweeper is played on a grid where some cells contain hidden mines.
2. Numbers on the grid represent how many mines are adjacent to that cell (including diagonally).
3. A cell with no number means it has no adjacent mines (this is represented as a blank cell).
4. The goal is to identify the location of all mines without detonating any.
5. You can mark a cell as containing a mine if you're certain based on logical deduction.
6. A mine location should be marked with `*`.
7. Cells that are empty (unrevealed cells) should be represented by a space (` `) character.
8. The final output should strictly follow the format provided below.

**Current Minesweeper State:**
The current state of the Minesweeper puzzle is shown in the image below.

In this representation:
- Numbers indicate the count of adjacent mines.
- Empty cells (unrevealed cells) are represented by a space (` `).
- The goal is to identify the positions of all the mines (using `*`).

**Output Format Requirements:**
1. Your final answer should mark all possible mine locations with `*` and **only `*`**..
3. Empty cells (unrevealed) should be represented by a space (` `),
and you should not place any numbers where the cells are blank.
4. Ensure that the output strictly follows the example format below:
   - Each row of the grid should be presented in the form: `|cell1|cell2|cell3|...|cellN|`
   - **No extra spaces should appear between cells or at the ends of rows.**
   - Each row must be terminated with a `\n` (newline) character.
5. **Do not add any additional spaces or empty lines in the answer.**
6. Follow the format shown below carefully:
   - Rows must consist of cells separated by the `|` character.
   - Every row must end with `\n` to ensure correct formatting.

**Example answer format:**
<answer>
|1|2|3|2|2|*|2|*|\n
|1|*|*|*|2|1|3|2|\n
|1|2|3|2|2|1|2|*|\n
| | | | |1|*|3|2|\n
| | |1|1|2|1|2|*|\n
| | |1|*|1| |1|1|\n
| | |1|2|2|1| | |\n
| | | |1|*|1| | |\n
</answer>

You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.

Now, think step by step, and provide your solution in the format shown above.
"""

PROMPT_EULERO_IMAGE = """
Your task is to solve the Eulero puzzle (also known as the Graeco-Latin Square or Euler Square),
based on the rules and the current puzzle state shown below.

**About the Puzzle**:
Eulero is a logic puzzle played on a square grid of size N×N.
Each cell must contain a **unique letter-number pair** (e.g., A1, B2).
It combines the logic of Latin squares and Greek squares.

**Goal**:
Fill all empty cells such that the following rules are satisfied:

**Global Rules (Graeco-Latin Square logic)**:
1. Each cell contains a **letter-number pair** (like A1).
2. Each **letter** appears **exactly once** in every row and every column.
3. Each **number** appears **exactly once** in every row and every column.
4. Each **letter-number pair** is **unique across the entire grid** (i.e., no duplicate pairs anywhere).

**Region Rules (Eulero-specific logic)**:
5. The grid is divided into **regions** of **exactly 3 cells each**, marked by thick black lines.
6. For each region:
   - Either all 3 cells must contain the **same letter-number pair** (e.g., all A1), OR
   - All 3 cells must contain **completely different** letter-number pairs (e.g., A1, B2, C3).
7. **Adjacent cells from different regions** (sharing an edge) must **not** contain the **same letter-number pair**.

**Grid Sizes**:
This puzzle can be of various sizes (e.g., 3×3, 4×4, 5×5, etc.).
The rules apply consistently across all sizes. The number of unique letters and numbers equals the grid size.

**Current Puzzle State**:
The puzzle is displayed in the image below:
- Some cells are pre-filled with letter-number pairs.
- Blank cells are empty and must be filled in.
- Thick black lines indicate the region boundaries.

**Your Output Format**:
1. final output must strictly follow this format:
   - Each row should be represented as a single line of **letter-number pairs**, separated by `|` (without spaces).
   - **Each row must be on a new line** using `\n` to separate them.

   **For example**:

   **For a 3×3 grid**:
   <answer>
   A1|B2|C3\nB3|C1|A2\nC2|A3|B1
   </answer>

   **For a 4×4 grid**:
   <answer>
   A1|B2|C3|D4\nB3|C4|D1|A2\nC2|D1|A4|B3\nD4|A3|B1|C2
   </answer>

   **For a 5×5 grid**:
   <answer>
   A1|B2|C3|D4|E5\nB3|C4|D1|E2|A5\nC2|D1|E4|A3|B5\nD4|E3|A2|B5|C1\nE5|A4|B1|C2|D3
   </answer>

   - **Do not add spaces between letter-number pairs**.
   - **Do not add any extra spaces or lines**.
   - **Make sure each row is separated by `\n`**.

You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.
Now, think step by step, and provide your solution in the format shown above.

"""

PROMPT_SNAKE_IMAGE = """You are a puzzle solver focusing on Snake puzzles (also known as Number Link or Tunnel puzzles).

In a Snake puzzle:
1. You need to find a path (snake) that connects the start point to the end point
2. The path must follow horizontal and vertical movements only (no diagonal moves)
3. The path cannot cross itself or branch out
4. The path must pass through exactly the number of cells in each row
and column as indicated by the row and column counts

The image shows a Snake puzzle. Analyze it to find:
- The grid size
- The start and end points
- The row and column counts
- The complete solution path

Find the complete path from start to end following the rules above.
Your answer should be a sequence of coordinates in the format (row,column) representing the path from start to end.

### Coordinate System:
- Use a 0-based coordinate system where (0,0) is the top-left cell of the grid
- Row numbers increase as you move downward
- Column numbers increase as you move rightward
- Coordinates are written as (row,column)

### Output Format Requirements:
Your answer should be a sequence of coordinates in the format (row,column) representing the path from start to end.
like: (row1,col1) (row2,col2) (row3,col3) ...

You should put your thinking process in `<think></think>` tags, and the final answer in `<answer></answer>` tags.
Now, think step by step, and provide your solution in the format shown above.
"""
