import re
from typing import Any, Dict, List

import numpy as np


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(
        self, predicted_answer: Any, ground_truth: Any,
        initial_state: str, params: Dict[str, Any] = None
    ) -> bool:
        raise NotImplementedError


class SokobanEvaluator(BaseEvaluator):
    # Define Sokoban elements for the simulator
    WALL = 1
    PLAYER = 2
    BOX = 3
    GOAL = 4
    BOX_ON_GOAL = 5
    PLAYER_ON_GOAL = 6
    FLOOR = 0

    # Directions mapping
    DIRECTIONS = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        from utils.constants import PROMPT_SOKOBAN
        return PROMPT_SOKOBAN.format(question)

    def extract_answer(self, model_output: str) -> List[str]:
        """
        Extract movement directions (up, down, left, right) from the model's output.
        Uses regex to extract valid directions regardless of formatting.
        """
        # Extract content within <answer></answer> tags if present
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, model_output, re.DOTALL)
        if match:
            content = match.group(1).strip().lower()
        else:
            content = model_output.strip().lower()

        # Define regex patterns for each direction
        # This will match directions even if surrounded by punctuation or other characters
        up_pattern = r'\b(?:up|u)\b'
        down_pattern = r'\b(?:down|d)\b'
        left_pattern = r'\b(?:left|l)\b'
        right_pattern = r'\b(?:right|r)\b'

        # Now extract directions in the order they appear in the text
        moves = []
        for word in re.findall(r'\b(?:up|u|down|d|left|l|right|r)\b', content):
            if re.match(up_pattern, word):
                moves.append('up')
            elif re.match(down_pattern, word):
                moves.append('down')
            elif re.match(left_pattern, word):
                moves.append('left')
            elif re.match(right_pattern, word):
                moves.append('right')

        return moves

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state) -> bool:
        """
        Robustly evaluates if the predicted answer solves the Sokoban puzzle.
        Can handle various formats of input for the predicted answer.
        """
        # If the predicted answer is a string, extract directions from it
        if isinstance(predicted_answer, str):
            predicted_moves = self.extract_answer(predicted_answer)
        # If it's already a list of directions, use it directly
        elif isinstance(predicted_answer, list):
            predicted_moves = predicted_answer
        else:
            return False

        # Parse the initial state into a grid
        if not initial_state or not isinstance(initial_state, str):
            return False

        lines = initial_state.strip().split('\n')
        height = len(lines)
        width = max(len(line) for line in lines)

        grid = np.zeros((height, width), dtype=int)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == '#':  # Wall
                    grid[y, x] = self.WALL
                elif char == '@':  # Player
                    grid[y, x] = self.PLAYER
                elif char == '$':  # Box
                    grid[y, x] = self.BOX
                elif char == '.':  # Target
                    grid[y, x] = self.GOAL
                elif char == '*':  # Box on target
                    grid[y, x] = self.BOX_ON_GOAL
                elif char == '+':  # Player on target
                    grid[y, x] = self.PLAYER_ON_GOAL

        # Create and use Sokoban simulator
        puzzle = self.SokobanSimulator(grid)

        # Apply each predicted move
        for move in predicted_moves:
            if not puzzle.move(move):
                return False  # Invalid move

        # Check if the puzzle is solved
        is_solved = puzzle.is_solved()

        # Debug: Check the ground truth solution if available
        if ground_truth and isinstance(ground_truth, str):
            # Get the list of moves from the ground truth solution
            gt_moves = self.extract_answer(ground_truth)

            # Create a new simulator to test the ground truth solution
            gt_puzzle = self.SokobanSimulator(grid.copy())

            # Apply each move from the ground truth solution
            for move in gt_moves:
                if not gt_puzzle.move(move):
                    break

        return is_solved

    # Define a Sokoban simulator class within the evaluator
    class SokobanSimulator:
        def __init__(self, grid):
            self.grid = grid.copy()
            self.height, self.width = grid.shape
            self.player_pos = None
            self.boxes = set()
            self.goals = set()

            # Define element constants from parent class
            self.WALL = 1
            self.PLAYER = 2
            self.BOX = 3
            self.GOAL = 4
            self.BOX_ON_GOAL = 5
            self.PLAYER_ON_GOAL = 6
            self.FLOOR = 0

            # Directions mapping
            self.DIRECTIONS = {
                'up': (-1, 0),
                'down': (1, 0),
                'left': (0, -1),
                'right': (0, 1)
            }

            # Find player, boxes, and goals
            for r in range(self.height):
                for c in range(self.width):
                    cell = self.grid[r, c]
                    if cell == self.PLAYER:
                        self.player_pos = (r, c)
                    elif cell == self.PLAYER_ON_GOAL:
                        self.player_pos = (r, c)
                        self.goals.add((r, c))
                    elif cell == self.BOX:
                        self.boxes.add((r, c))
                    elif cell == self.BOX_ON_GOAL:
                        self.boxes.add((r, c))
                        self.goals.add((r, c))
                    elif cell == self.GOAL:
                        self.goals.add((r, c))

        def move(self, direction):
            """Move the player in the given direction if possible."""
            if direction.lower() not in self.DIRECTIONS:
                return False

            dr, dc = self.DIRECTIONS[direction.lower()]
            r, c = self.player_pos
            new_r, new_c = r + dr, c + dc

            # Check bounds
            if not (0 <= new_r < self.height and 0 <= new_c < self.width):
                return False

            # Check if moving into a wall
            if self.grid[new_r, new_c] == self.WALL:
                return False

            # Check if moving into a box
            if (new_r, new_c) in self.boxes:
                # Calculate position behind the box
                box_r, box_c = new_r + dr, new_c + dc

                # Check bounds and if box can be pushed
                if not (0 <= box_r < self.height and 0 <= box_c < self.width):
                    return False

                # Check if box destination is valid (not a wall or another box)
                if self.grid[box_r, box_c] == self.WALL or (box_r, box_c) in self.boxes:
                    return False

                # Move the box
                self.boxes.remove((new_r, new_c))
                self.boxes.add((box_r, box_c))

            # Move the player
            self.player_pos = (new_r, new_c)
            return True

        def is_solved(self):
            """检查所有箱子是否都在目标点上"""
            # 检查所有箱子是否都在目标点上，以及所有目标点是否都有箱子
            boxes_on_goals = all(box in self.goals for box in self.boxes)
            goals_with_boxes = all(goal in self.boxes for goal in self.goals)
            solved = boxes_on_goals and goals_with_boxes

            return solved
