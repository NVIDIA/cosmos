import ast
import json
import re
from typing import Any, Dict, List


class FutoshikiEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """Format the question for the model."""
        prompt = (
            "Solve the Futoshiki puzzle described below. A Futoshiki puzzle uses a grid where:\n"
            "1. Each row and column must contain each number exactly once (like Sudoku)\n"
            "2. Inequality signs (< and >) between cells must be satisfied\n\n"
            f"{question}\n\n"
            "Provide your answer as a grid of numbers. Format your answer as a list of lists, "
            "where each inner list represents a row of the grid."
        )
        return prompt

    def extract_answer(self, model_output: str) -> List[List[int]]:
        """Extract the grid solution from the model's output with enhanced robustness."""
        if not isinstance(model_output, str):
            model_output = str(model_output)

        # Method 1: Try to find and parse nested list structures
        grid = self._extract_nested_lists(model_output)
        if grid:
            return grid

        # Method 2: Try to parse as Python literal
        grid = self._extract_python_literal(model_output)
        if grid:
            return grid

        # Method 3: Extract from table-like format
        grid = self._extract_table_format(model_output)
        if grid:
            return grid

        # Method 4: Extract from comma/space separated format
        grid = self._extract_separated_format(model_output)
        if grid:
            return grid

        # Method 5: Extract all numbers and try to infer grid structure
        grid = self._extract_inferred_grid(model_output)
        if grid:
            return grid

        # Method 6: Handle single line formats
        grid = self._extract_single_line_format(model_output)
        if grid:
            return grid

        return []

    def _extract_nested_lists(self, text: str) -> List[List[int]]:
        """Extract nested list structures like [[1,2,3],[4,5,6]]"""
        # Enhanced pattern to match various nested list formats
        patterns = [
            r'\[\s*(?:\[\s*(?:\d+(?:\s*,\s*\d+)*)\s*\](?:\s*,\s*)?)+\s*\]',  # Standard format
            r'\[(?:\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*)?)+\s*\]',      # Compact format
            r'\[\s*\[[\d\s,]+\](?:\s*,\s*\[[\d\s,]+\])*\s*\]'                # Flexible spacing
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    grid = ast.literal_eval(match)
                    if self._is_valid_grid(grid):
                        return grid
                except Exception:
                    continue
        return None

    def _extract_python_literal(self, text: str) -> List[List[int]]:
        """Try to extract and evaluate Python literal expressions"""
        # Find potential list structures
        bracket_matches = []
        stack = []
        start = -1

        for i, char in enumerate(text):
            if char == '[':
                if not stack:
                    start = i
                stack.append(char)
            elif char == ']':
                if stack:
                    stack.pop()
                    if not stack and start != -1:
                        bracket_matches.append(text[start:i + 1])

        # Try to evaluate each match
        for match in bracket_matches:
            try:
                grid = ast.literal_eval(match)
                if self._is_valid_grid(grid):
                    return grid
            except Exception:
                continue

        return None

    def _extract_table_format(self, text: str) -> List[List[int]]:
        """Extract from table-like formats with | or other separators"""
        lines = text.strip().split('\n')
        grid = []

        for line in lines:
            # Skip empty lines and lines without numbers
            if not line.strip():
                continue

            # Handle different table formats
            row_numbers = []

            # Try different separators
            for separator in ['|', '\t', '  ', ' ']:
                if separator in line:
                    parts = line.split(separator)
                    numbers = []
                    for part in parts:
                        nums = re.findall(r'\d+', part.strip())
                        numbers.extend([int(n) for n in nums])
                    if numbers:
                        row_numbers = numbers
                        break

            # If no separator worked, extract all numbers from the line
            if not row_numbers:
                row_numbers = [int(n) for n in re.findall(r'\d+', line)]

            if row_numbers:
                grid.append(row_numbers)

        if self._is_valid_grid(grid):
            return grid

        return None

    def _extract_separated_format(self, text: str) -> List[List[int]]:
        """Extract from comma or space separated values"""
        lines = text.strip().split('\n')
        grid = []

        for line in lines:
            # Skip lines that don't contain numbers
            if not re.search(r'\d', line):
                continue

            # Remove brackets and extract numbers
            clean_line = re.sub(r'[\[\]()]', '', line)

            # Try comma separation first
            if ',' in clean_line:
                numbers = []
                for part in clean_line.split(','):
                    nums = re.findall(r'\d+', part)
                    numbers.extend([int(n) for n in nums])
            else:
                # Space separation
                numbers = [int(n) for n in re.findall(r'\d+', clean_line)]

            if numbers:
                grid.append(numbers)

        if self._is_valid_grid(grid):
            return grid

        return None

    def _extract_single_line_format(self, text: str) -> List[List[int]]:
        """Extract from single line formats like '1 2 3 4 5 1 2 3 4 5 ...'"""
        # Extract all numbers from the text
        all_numbers = [int(n) for n in re.findall(r'\d+', text)]

        if not all_numbers:
            return None

        # Try different grid sizes
        possible_sizes = []
        total_numbers = len(all_numbers)

        # Check perfect squares
        for size in range(2, int(total_numbers**0.5) + 2):
            if size * size == total_numbers:
                possible_sizes.append(size)

        # Check rectangles
        for rows in range(2, total_numbers // 2 + 1):
            if total_numbers % rows == 0:
                cols = total_numbers // rows
                if cols >= 2:
                    possible_sizes.append((rows, cols))

        # Try each possible size
        for size in possible_sizes:
            if isinstance(size, int):
                # Square grid
                grid = []
                for i in range(size):
                    grid.append(all_numbers[i * size:(i + 1) * size])
                if self._is_valid_grid(grid):
                    return grid
            else:
                # Rectangular grid
                rows, cols = size
                grid = []
                for i in range(rows):
                    grid.append(all_numbers[i * cols:(i + 1) * cols])
                if self._is_valid_grid(grid):
                    return grid

        return None

    def _extract_inferred_grid(self, text: str) -> List[List[int]]:
        """Extract numbers and try to infer the grid structure from context"""
        # Look for hints about grid size in the text
        size_hints = re.findall(r'(\d+)\s*[×x]\s*(\d+)', text)
        if size_hints:
            rows, cols = int(size_hints[0][0]), int(size_hints[0][1])
        else:
            # Look for patterns that suggest grid size
            lines_with_numbers = []
            for line in text.split('\n'):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    lines_with_numbers.append(len(numbers))

            if lines_with_numbers:
                # Use the most common line length as columns
                cols = max(set(lines_with_numbers), key=lines_with_numbers.count)
                rows = len(lines_with_numbers)
            else:
                # Extract all numbers and assume square grid
                all_numbers = [int(n) for n in re.findall(r'\d+', text)]
                if not all_numbers:
                    return None

                size = int(len(all_numbers) ** 0.5)
                if size * size == len(all_numbers):
                    rows = cols = size
                else:
                    return None

        # Extract all numbers and arrange in grid
        all_numbers = [int(n) for n in re.findall(r'\d+', text)]
        if len(all_numbers) != rows * cols:
            return None

        grid = []
        for i in range(rows):
            grid.append(all_numbers[i * cols:(i + 1) * cols])

        if self._is_valid_grid(grid):
            return grid

        return None

    def _is_valid_grid(self, grid) -> bool:
        """Check if the extracted grid is valid"""
        if not isinstance(grid, list) or not grid:
            return False

        if not all(isinstance(row, list) for row in grid):
            return False

        if not all(len(row) == len(grid[0]) for row in grid):
            return False

        # Check if all elements are integers
        try:
            for row in grid:
                for cell in row:
                    int(cell)
        except (ValueError, TypeError):
            return False

        # Grid should be at least 2x2
        if len(grid) < 2 or len(grid[0]) < 2:
            return False

        return True

    def _normalize_grid(self, answer_str_or_list):
        """Normalize the input to a grid of integers."""
        if isinstance(answer_str_or_list, list):
            # If it's already a list, check if it's a valid grid
            if all(isinstance(row, list) for row in answer_str_or_list):
                return answer_str_or_list
            return None

        # If it's a string, use extract_answer method
        return self.extract_answer(str(answer_str_or_list))

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Dict[str, Any]) -> bool:
        """
        Evaluate the predicted answer based on Futoshiki game rules and initial state.

        Args:
            predicted_answer: The model's output (string or list)
            ground_truth: The ground truth solution (not used in rule-based evaluation)
            initial_state: Dictionary containing:
                - 'grid': 2D list with initial numbers (0 for empty cells)
                - 'inequalities': List of inequality constraints
                - 'size': Grid size

        Returns:
            bool: True if the predicted answer satisfies all Futoshiki rules
        """
        try:
            # Extract the predicted grid using robust parsing
            if isinstance(predicted_answer, str):
                predicted_grid = self.extract_answer(predicted_answer)
            elif isinstance(predicted_answer, list):
                if self._is_valid_grid(predicted_answer):
                    predicted_grid = predicted_answer
                else:
                    return False
            else:
                predicted_grid = self.extract_answer(str(predicted_answer))

            # If extraction failed, return False
            if not predicted_grid:
                return False

            # Parse initial state (support dict or JSON/string input)
            if not isinstance(initial_state, dict):
                try:
                    if isinstance(initial_state, str):
                        try:
                            initial_state = json.loads(initial_state)
                        except json.JSONDecodeError:
                            # Fallback to Python literal (handles single quotes)
                            initial_state = ast.literal_eval(initial_state)
                    else:
                        # Unsupported type
                        return False
                except Exception:
                    return False

            initial_grid = initial_state.get('grid', [])
            inequalities = initial_state.get('inequalities', [])
            expected_size = initial_state.get('size', len(initial_grid))

            # Validate grid dimensions
            if len(predicted_grid) != expected_size:
                return False

            if any(len(row) != expected_size for row in predicted_grid):
                return False

            # Convert to integers and validate range
            try:
                predicted_grid = [[int(cell) for cell in row] for row in predicted_grid]
            except (ValueError, TypeError):
                return False

            # Check if all numbers are in valid range [1, n]
            for row in predicted_grid:
                for cell in row:
                    if cell < 1 or cell > expected_size:
                        return False

            # Rule 1: Check if predicted grid preserves initial numbers
            if not self._check_initial_numbers(predicted_grid, initial_grid):
                return False

            # Rule 2: Check if each row contains each number exactly once
            if not self._check_rows_unique(predicted_grid):
                return False

            # Rule 3: Check if each column contains each number exactly once
            if not self._check_columns_unique(predicted_grid):
                return False

            # Rule 4: Check if all inequality constraints are satisfied
            if not self._check_inequalities(predicted_grid, inequalities):
                return False

            return True

        except Exception:
            return False

    def _check_initial_numbers(self, predicted_grid: List[List[int]], initial_grid: List[List[int]]) -> bool:
        """Check if the predicted grid preserves all initial numbers."""
        if len(predicted_grid) != len(initial_grid):
            return False

        for i in range(len(initial_grid)):
            if len(predicted_grid[i]) != len(initial_grid[i]):
                return False
            for j in range(len(initial_grid[i])):
                # If there was an initial number (not 0), it must be preserved
                if initial_grid[i][j] != 0 and predicted_grid[i][j] != initial_grid[i][j]:
                    return False

        return True

    def _check_rows_unique(self, grid: List[List[int]]) -> bool:
        """Check if each row contains each number from 1 to n exactly once."""
        n = len(grid)
        expected_set = set(range(1, n + 1))

        for row in grid:
            if set(row) != expected_set:
                return False

        return True

    def _check_columns_unique(self, grid: List[List[int]]) -> bool:
        """Check if each column contains each number from 1 to n exactly once."""
        n = len(grid)
        expected_set = set(range(1, n + 1))

        for j in range(n):
            column = [grid[i][j] for i in range(n)]
            if set(column) != expected_set:
                return False

        return True

    def _check_inequalities(self, grid: List[List[int]], inequalities: List[Dict]) -> bool:
        """Check if all inequality constraints are satisfied."""
        for inequality in inequalities:
            try:
                # Get cell positions
                cell1 = inequality['cell1']
                cell2 = inequality['cell2']
                symbol = inequality['symbol']

                # Get cell values
                i1, j1 = cell1[0], cell1[1]
                i2, j2 = cell2[0], cell2[1]

                value1 = grid[i1][j1]
                value2 = grid[i2][j2]

                # Check inequality constraint
                if symbol == '<':
                    if value1 >= value2:
                        return False
                elif symbol == '>':
                    if value1 <= value2:
                        return False
                else:
                    # Invalid symbol
                    return False

            except (KeyError, IndexError, TypeError):
                # Invalid inequality format
                return False

        return True

    # Legacy method for backward compatibility
    def old_evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state=None) -> bool:
        """
        Legacy evaluation method that compares predicted answer with ground truth.
        """
        # Parse ground truth
        try:
            if isinstance(ground_truth, str):
                gt_grid = ast.literal_eval(ground_truth)
            else:
                gt_grid = ground_truth
        except (SyntaxError, ValueError):
            return False

        if not isinstance(gt_grid, list) or not all(isinstance(row, list) for row in gt_grid):
            return False

        # Extract predicted answer using the robust extract_answer method
        if isinstance(predicted_answer, str):
            predicted_grid = self.extract_answer(predicted_answer)
        elif isinstance(predicted_answer, list):
            # If it's already a list, validate it
            if self._is_valid_grid(predicted_answer):
                predicted_grid = predicted_answer
            else:
                predicted_grid = []
        else:
            # Convert to string and extract
            predicted_grid = self.extract_answer(str(predicted_answer))

        # If extraction failed, return False
        if not predicted_grid:
            return False

        # Check dimensions
        if len(predicted_grid) != len(gt_grid):
            return False

        if any(len(row) != len(gt_grid[0]) for row in predicted_grid):
            return False

        # Compare grids element by element
        try:
            for i in range(len(gt_grid)):
                for j in range(len(gt_grid[0])):
                    if int(predicted_grid[i][j]) != int(gt_grid[i][j]):
                        return False
        except (ValueError, TypeError, IndexError):
            return False

        return True
