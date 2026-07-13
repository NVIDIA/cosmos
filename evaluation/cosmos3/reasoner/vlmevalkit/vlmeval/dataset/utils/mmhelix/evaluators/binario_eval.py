import re
from typing import Any, Dict, List, Optional


class BaseEvaluator:

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any = None,
                 initial_state: List[List[Any]] = None) -> bool:
        raise NotImplementedError


class BinarioEvaluator(BaseEvaluator):
    """
    Enhanced evaluator for the Binairo task.
    This task involves evaluating a matrix of 0s and 1s with improved robustness.
    """

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """
        Prepares the prompt for the model.

        Args:
            question: The question to be asked
            params: Additional parameters for customizing the prompt

        Returns:
            The prepared prompt string
        """
        return question

    def extract_answer(self, model_output: str) -> Optional[List[List[int]]]:
        """
        Enhanced extraction of matrix from model output with improved robustness.

        Args:
            model_output: The raw output from the model

        Returns:
            A list of lists representing the matrix, or None if no valid matrix is found
        """
        if not model_output or not isinstance(model_output, str):
            return None

        # Clean the output
        output = model_output.strip()

        # Multiple extraction strategies, ordered by preference
        strategies = [
            self._extract_from_literal_newlines,      # Handle \n literals
            self._extract_from_code_blocks,
            self._extract_from_answer_section,
            self._extract_from_solution_section,
            self._extract_from_result_section,
            self._extract_from_matrix_keywords,       # Enhanced matrix keyword detection
            self._extract_matrix_patterns,
            self._extract_from_table_format,
            self._extract_from_bracket_format,        # [1,0,1] format
            self._extract_from_space_separated,       # Space-separated numbers
            self._extract_from_json_like,             # JSON-like format
            self._extract_from_numeric_blocks,        # Pure numeric blocks
            self._extract_from_grid_format,           # Grid-like formats
            self._extract_any_grid_pattern,
            self._extract_from_single_line,           # All numbers in one line
            self._extract_from_mixed_format,          # Mixed formats
            self._extract_from_coordinate_format,     # R1C1=0 format
            self._extract_from_quoted_strings,        # "0 1 0\n1 0 1" format
            self._extract_from_enumerated_format,     # 1. 0 1 0, 2. 1 0 1 format
        ]

        for strategy in strategies:
            matrix = strategy(output)
            if matrix is not None and self._is_valid_matrix(matrix):
                return matrix

        return None

    def _extract_from_literal_newlines(self, output: str) -> Optional[List[List[int]]]:
        """Extract matrix from strings with literal \\n characters - enhanced version"""
        # Handle various literal newline representations
        patterns = [
            r'([01](?:\s+[01])+(?:\\n[01](?:\s+[01])+)*)',  # Numbers with literal \n
            r'([01](?:[,\s]+[01])+(?:\\n[01](?:[,\s]+[01])+)*)',  # With commas
            r'([01](?:[,\s]+[01])+(?:[\s]*\\n[\s]*[01](?:[,\s]+[01])+)*)',  # More flexible spacing
            r'([01](?:\s*[01])+(?:\\n[01](?:\s*[01])+)*)',  # Tight spacing
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, output)
            for match in matches:
                matrix_str = match.group(1)
                # Handle multiple literal newline formats
                for nl_format in ['\\n', '\\r\\n', '\\r']:
                    matrix_str = matrix_str.replace(nl_format, '\n')

                matrix = self._parse_matrix_string(matrix_str)
                if matrix is not None:
                    return matrix

        # Also check for the exact format with various newline representations
        for nl_format in ['\\n', '\\r\\n', '\\r']:
            if nl_format in output:
                cleaned_output = output.replace(nl_format, '\n')
                matrix = self._parse_matrix_string(cleaned_output)
                if matrix is not None:
                    return matrix

        return None

    def _extract_from_code_blocks(self, output: str) -> Optional[List[List[int]]]:
        """Extract matrix from code blocks (```...```)"""
        patterns = [
            r'```(?:python|text|matrix|grid|answer|solution)?\s*\n?([\d\s\n,|\t\\]+?)```',
            r'```\s*([\d\s\n,|\t\\]+?)\s*```',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, output, re.MULTILINE | re.DOTALL)
            for match in matches:
                content = match.group(1).replace('\\n', '\n')
                matrix = self._parse_matrix_string(content)
                if matrix is not None:
                    return matrix
        return None

    def _extract_from_answer_section(self, output: str) -> Optional[List[List[int]]]:
        """Extract matrix from answer sections - enhanced version"""
        patterns = [
            # Standard answer patterns
            r'(?:answer|solution|result):\s*([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',
            r'(?:answer|solution|result)\s*[=:]\s*([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',
            r'(?:final\s+)?(?:answer|solution|result)\s*[=:]?\s*\n?([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',

            # Enhanced patterns with more keywords
            r'(?:my\s+)?(?:answer|solution|result)\s*[=:]?\s*\n?([01][\d\s\n,|\t\\]+?)'
            r'(?:\n\s*\n|\Z)',
            r'(?:the\s+)?(?:correct\s+)?(?:answer|solution|result)\s*[=:]?\s*\n?'
            r'([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',
            r'(?:here\s+is\s+)?(?:my\s+)?(?:answer|solution|result)\s*[=:]?\s*\n?'
            r'([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',

            # Pattern for answers at the end of text
            r'(?:answer|solution|result).*?([01](?:\s+[01])+(?:[\\n\n][01](?:\s+[01])+)*)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                content = match.group(1).replace('\\n', '\n')
                matrix = self._parse_matrix_string(content)
                if matrix is not None:
                    return matrix
        return None

    def _extract_from_solution_section(self, output: str) -> Optional[List[List[int]]]:
        """Extract matrix from solution sections"""
        patterns = [
            r'(?:the\s+)?(?:solution|matrix|grid)\s+is:?\s*\n?([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',
            r'(?:completed|filled|final)\s+(?:matrix|grid):?\s*\n?([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',
            r'(?:solution|matrix|grid).*?([01](?:\s+[01])+(?:[\\n\n][01](?:\s+[01])+)*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                content = match.group(1).replace('\\n', '\n')
                matrix = self._parse_matrix_string(content)
                if matrix is not None:
                    return matrix
        return None

    def _extract_from_result_section(self, output: str) -> Optional[List[List[int]]]:
        """Extract matrix from result sections"""
        patterns = [
            r'(?:here\s+is\s+the\s+)?(?:result|output):?\s*\n?([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',
            r'(?:the\s+)?(?:final|complete)\s+(?:result|output):?\s*\n?([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                content = match.group(1).replace('\\n', '\n')
                matrix = self._parse_matrix_string(content)
                if matrix is not None:
                    return matrix
        return None

    def _extract_from_bracket_format(self, output: str) -> Optional[List[List[int]]]:
        """Extract matrix from bracket formats like [1,0,1] or [[1,0,1],[0,1,0]]"""
        patterns = [
            # [[1,0,1],[0,1,0]]
            r'\[\s*\[\s*([01](?:\s*,\s*[01])*)\s*\](?:\s*,\s*\[\s*([01](?:\s*,\s*[01])*)\s*\])*\s*\]',
            # [1,0,1]\n[0,1,0]
            r'(?:\[\s*([01](?:\s*,\s*[01])*)\s*\](?:\s*,?\s*\n?)){2,}',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, output)
            for match in matches:
                # Extract all bracket contents
                bracket_contents = re.findall(r'\[\s*([01](?:\s*,\s*[01])*)\s*\]', match.group(0))
                if len(bracket_contents) >= 2:
                    matrix = []
                    for content in bracket_contents:
                        row = [int(x.strip()) for x in content.split(',') if x.strip().isdigit()]
                        if all(cell in [0, 1] for cell in row) and len(row) > 0:
                            matrix.append(row)

                    if len(matrix) >= 2 and self._is_valid_matrix(matrix):
                        return matrix

        return None

    def _extract_from_space_separated(self, output: str) -> Optional[List[List[int]]]:
        """Extract from simple space-separated format - enhanced version"""
        # Split by various line separators
        lines = re.split(r'[\n\r]+', output)
        matrix = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Enhanced pattern matching for binary rows
            # Check if line contains primarily 0s, 1s, and separators
            if re.match(r'^[01\s,\|\-\+\.\t]*$', line):
                # Extract only 0s and 1s
                binary_chars = re.findall(r'[01]', line)
                if len(binary_chars) >= 2:  # At least 2 binary digits
                    row = [int(char) for char in binary_chars]
                    matrix.append(row)

        if len(matrix) >= 2 and self._is_valid_matrix(matrix):
            return matrix

        return None

    def _extract_from_json_like(self, output: str) -> Optional[List[List[int]]]:
        """Extract from JSON-like formats"""
        patterns = [
            r'\[\s*\[([01](?:\s*,\s*[01])*)\](?:\s*,\s*\[([01](?:\s*,\s*[01])*)\])*\s*\]',
            r'(?:matrix|grid|answer)\s*[:=]\s*(\[\s*\[[\d\s,\]]+\])',
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                try:
                    # Try to evaluate as Python list
                    content = match.group(1) if len(match.groups()) >= 1 else match.group(0)
                    # Clean and make it safe for eval
                    content = re.sub(r'[^\[\]01,\s]', '', content)
                    matrix = eval(content)
                    if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
                        if self._is_valid_matrix(matrix):
                            return matrix
                except Exception:
                    pass

        return None

    def _extract_from_mixed_format(self, output: str) -> Optional[List[List[int]]]:
        """Extract from mixed formats with separators like | or tabs"""
        patterns = [
            r'([01](?:[\s|,\t]+[01])+(?:\n[01](?:[\s|,\t]+[01])+)*)',  # Mixed separators
            r'([01](?:[|]+[01])+(?:\n[01](?:[|]+[01])+)*)',            # Pipe separated
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, output, re.MULTILINE)
            for match in matches:
                content = match.group(1)
                matrix = self._parse_matrix_string(content)
                if matrix is not None:
                    return matrix
        return None

    def _extract_from_coordinate_format(self, output: str) -> Optional[List[List[int]]]:
        """Extract from coordinate-based format like R1C1=0, R1C2=1"""
        # Look for patterns like R1C1=0, R1C2=1, etc.
        coord_pattern = r'R(\d+)C(\d+)\s*[=:]\s*([01])'
        matches = re.findall(coord_pattern, output, re.IGNORECASE)

        if matches:
            # Determine grid size
            max_row = max(int(m[0]) for m in matches)
            max_col = max(int(m[1]) for m in matches)

            # Check if we have enough coordinates for a complete grid
            if len(matches) >= max_row * max_col * 0.5:  # At least half filled
                matrix = [[None for _ in range(max_col)] for _ in range(max_row)]

                for row_str, col_str, val_str in matches:
                    row_idx = int(row_str) - 1  # Convert to 0-based
                    col_idx = int(col_str) - 1
                    if 0 <= row_idx < max_row and 0 <= col_idx < max_col:
                        matrix[row_idx][col_idx] = int(val_str)

                # Convert None to appropriate values if the grid is mostly filled
                filled_count = sum(1 for row in matrix for cell in row if cell is not None)
                if filled_count >= max_row * max_col * 0.8:  # 80% filled
                    # This might be a complete solution, return as is
                    return [[cell if cell is not None else 0 for cell in row] for row in matrix]

        return None

    def _extract_from_single_line(self, output: str) -> Optional[List[List[int]]]:
        """Extract from single line with all numbers"""
        # Look for long sequences of 0s and 1s that might represent a flattened matrix
        pattern = r'([01](?:[,\s]*[01]){5,})'  # At least 6 numbers (could be 2x3 or 3x2 etc.)

        matches = re.finditer(pattern, output)
        for match in matches:
            numbers_str = match.group(1)
            numbers = [int(x) for x in re.findall(r'[01]', numbers_str)]

            if len(numbers) >= 4:  # At least 2x2
                # Try different matrix dimensions
                for rows in range(2, int(len(numbers)**0.5) + 2):
                    if len(numbers) % rows == 0:
                        cols = len(numbers) // rows
                        if cols >= 2:  # At least 2 columns
                            matrix = []
                            for i in range(rows):
                                row = numbers[i * cols:(i + 1) * cols]
                                matrix.append(row)

                            if self._is_valid_matrix(matrix):
                                return matrix

        return None

    def _extract_matrix_patterns(self, output: str) -> Optional[List[List[int]]]:
        """Extract using general matrix patterns"""
        # Look for consecutive lines that look like matrix rows
        lines = output.replace('\\n', '\n').split('\n')
        matrix_candidates = []
        current_matrix = []

        for line in lines:
            line = line.strip()
            # Check if line looks like a matrix row
            if self._is_matrix_row(line):
                row = self._parse_matrix_row(line)
                if row is not None:
                    current_matrix.append(row)
                else:
                    if len(current_matrix) >= 2:  # At least 2 rows to be considered a matrix
                        matrix_candidates.append(current_matrix[:])
                    current_matrix = []
            else:
                if len(current_matrix) >= 2:
                    matrix_candidates.append(current_matrix[:])
                current_matrix = []

        # Don't forget the last matrix if it exists
        if len(current_matrix) >= 2:
            matrix_candidates.append(current_matrix)

        # Return the largest valid matrix
        for matrix in sorted(matrix_candidates, key=len, reverse=True):
            if self._is_valid_matrix(matrix):
                return matrix

        return None

    def _extract_from_table_format(self, output: str) -> Optional[List[List[int]]]:
        """Extract from table-like formats with |"""
        patterns = [
            r'\|[\d\s\|]+\|',  # |0 1 0|
            r'[\d\s]+\|[\d\s\|]+',  # 0 1 0|1 0 1|
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.MULTILINE)
            if matches:
                matrix = []
                for match in matches:
                    # Clean up the match and extract numbers
                    cleaned = re.sub(r'[|\s]+', ' ', match).strip()
                    row = self._parse_matrix_row(cleaned)
                    if row is not None:
                        matrix.append(row)

                if len(matrix) >= 2 and self._is_valid_matrix(matrix):
                    return matrix

        return None

    def _extract_any_grid_pattern(self, output: str) -> Optional[List[List[int]]]:
        """Last resort: extract any pattern that looks like a grid"""
        # Very permissive pattern for any sequence of 0s and 1s that could form a grid
        output = output.replace('\\n', '\n')
        pattern = r'(?:^|\n)\s*([01](?:[\s,|\t]+[01]){1,20})\s*(?:\n|$)'
        matches = re.findall(pattern, output, re.MULTILINE)

        if len(matches) >= 2:
            matrix = []
            for match in matches:
                row = self._parse_matrix_row(match)
                if row is not None and len(row) > 1:  # At least 2 elements per row
                    matrix.append(row)

            if len(matrix) >= 2 and self._is_valid_matrix(matrix):
                return matrix

        return None

    def _is_matrix_row(self, line: str) -> bool:
        """Check if a line looks like a matrix row"""
        if not line.strip():
            return False

        # Should contain only digits, spaces, commas, tabs, and pipes
        if not re.match(r'^[\d\s,|\t]+$', line.strip()):
            return False

        # Should have at least 2 numbers
        numbers = re.findall(r'\d+', line)
        return len(numbers) >= 2 and all(num in ['0', '1'] for num in numbers)

    def _parse_matrix_row(self, row_string: str) -> Optional[List[int]]:
        """Parse a single row string into a list of integers with enhanced robustness"""
        if not row_string:
            return None

        # Clean the string and extract numbers
        cleaned = re.sub(r'[,|\t\-\+\=]+', ' ', row_string)  # Replace various separators with spaces
        cleaned = re.sub(r'[^\d\s]', ' ', cleaned)  # Remove any non-digit, non-space characters
        numbers = cleaned.strip().split()

        try:
            row = []
            for num_str in numbers:
                if num_str.isdigit():
                    num = int(num_str)
                    if num in [0, 1]:
                        row.append(num)
                    # Skip invalid numbers (not 0 or 1)

            # Return only if we have at least 2 valid binary digits
            if len(row) >= 2:
                return row
        except (ValueError, TypeError):
            pass

        return None

    def _parse_matrix_string(self, matrix_string: str) -> Optional[List[List[int]]]:
        """Parse a multi-line matrix string with enhanced robustness"""
        if not matrix_string:
            return None

        matrix = []
        # Replace various newline representations
        matrix_string = matrix_string.replace('\\n', '\n').replace('\\r\\n', '\n').replace('\\r', '\n')

        # Split by various line separators
        rows = re.split(r'[\n\r]+', matrix_string.strip())

        expected_row_length = None
        for row_str in rows:
            row_str = row_str.strip()
            if not row_str:  # Skip empty rows
                continue

            row = self._parse_matrix_row(row_str)
            if row is not None and len(row) > 0:
                # Ensure consistent row length
                if expected_row_length is None:
                    expected_row_length = len(row)
                elif len(row) != expected_row_length:
                    # Skip rows with inconsistent length
                    continue

                matrix.append(row)

        return matrix if len(matrix) >= 2 else None

    def _is_valid_matrix(self, matrix: List[List[int]]) -> bool:
        """Validate that the matrix is well-formed - enhanced version"""
        if not matrix or not isinstance(matrix, list):
            return False

        if len(matrix) == 0:
            return False

        # Check that all rows have the same length
        row_length = len(matrix[0])
        if row_length == 0:
            return False

        for row in matrix:
            if not isinstance(row, list) or len(row) != row_length:
                return False

            # Check that all elements are valid (0 or 1)
            for cell in row:
                if not isinstance(cell, int) or cell not in [0, 1]:
                    return False

        # Matrix should be at least 2x2 and dimensions should make sense for Binairo
        if len(matrix) < 2 or row_length < 2:
            return False

        # For Binairo, dimensions should typically be even (though not strictly required)
        # This is a soft check - we'll accept odd dimensions but prefer even
        return True

    def _extract_from_matrix_keywords(self, output: str) -> Optional[List[List[int]]]:
        """Enhanced extraction from text with matrix-related keywords"""
        # More comprehensive patterns for matrix keywords
        patterns = [
            r'(?:the\s+)?(?:completed|final|solved|answer)\s+(?:matrix|grid|puzzle)'
            r'\s*[:\-]?\s*\n?([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|$)',
            r'(?:matrix|grid|puzzle)\s+(?:solution|answer|result)\s*[:\-]?\s*\n?'
            r'([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|$)',
            r'(?:here\s+is\s+the\s+)?(?:matrix|grid|solution|answer)\s*[:\-]?\s*\n'
            r'([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|$)',
            r'(?:binairo|takuzu)\s+(?:solution|answer)\s*[:\-]?\s*\n?'
            r'([01][\d\s\n,|\t\\]+?)(?:\n\s*\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                content = match.group(1).replace('\\n', '\n')
                matrix = self._parse_matrix_string(content)
                if matrix is not None:
                    return matrix
        return None

    def _extract_from_numeric_blocks(self, output: str) -> Optional[List[List[int]]]:
        """Extract from blocks of numbers separated by clear delimiters"""
        # Look for blocks of binary digits with clear separations
        patterns = [
            r'(?:^|\n)\s*([01](?:[,\s]+[01])+)\s*(?:\n|$)',  # Simple space/comma separated
            r'(?:^|\n)\s*([01](?:\s*[01])+)\s*(?:\n|$)',     # Tightly packed digits
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.MULTILINE)
            if len(matches) >= 2:
                matrix = []
                for match in matches:
                    row = self._parse_matrix_row(match)
                    if row is not None and len(row) >= 2:
                        matrix.append(row)

                if len(matrix) >= 2 and self._is_valid_matrix(matrix):
                    return matrix
        return None

    def _extract_from_grid_format(self, output: str) -> Optional[List[List[int]]]:
        """Extract from grid-like visual formats"""
        # Handle ASCII grid formats with borders
        patterns = [
            r'(?:\+[\-\+]+\+\s*\n)([01\s\|]+)(?:\n\+[\-\+]+\+)?',  # +---+ bordered grids
            r'(?:\|[01\s\|]+\|\s*\n){2,}',                          # Simple | delimited rows
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, output, re.MULTILINE)
            for match in matches:
                content = match.group(0)
                # Extract rows from the grid
                rows = re.findall(r'\|([01\s]+)\|', content)
                if len(rows) >= 2:
                    matrix = []
                    for row_str in rows:
                        row = self._parse_matrix_row(row_str)
                        if row is not None:
                            matrix.append(row)

                    if len(matrix) >= 2 and self._is_valid_matrix(matrix):
                        return matrix
        return None

    def _extract_from_quoted_strings(self, output: str) -> Optional[List[List[int]]]:
        """Extract from quoted string formats"""
        # Handle quoted matrix strings
        patterns = [
            r'"([01](?:[\s,]+[01])+(?:[\\n\n][01](?:[\s,]+[01])+)*)"',      # Double quotes
            r"'([01](?:[\s,]+[01])+(?:[\\n\n][01](?:[\s,]+[01])+)*)'",      # Single quotes
            r'`([01](?:[\s,]+[01])+(?:[\\n\n][01](?:[\s,]+[01])+)*)`',      # Backticks
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                content = match.group(1).replace('\\n', '\n')
                matrix = self._parse_matrix_string(content)
                if matrix is not None:
                    return matrix
        return None

    def _extract_from_enumerated_format(self, output: str) -> Optional[List[List[int]]]:
        """Extract from enumerated row formats like '1. 0 1 0', '2. 1 0 1'"""
        # Handle numbered rows
        pattern = r'(?:^|\n)\s*\d+[\.\)]\s*([01](?:[\s,]+[01])+)\s*(?:\n|$)'
        matches = re.findall(pattern, output, re.MULTILINE)

        if len(matches) >= 2:
            matrix = []
            for match in matches:
                row = self._parse_matrix_row(match)
                if row is not None:
                    matrix.append(row)

            if len(matrix) >= 2 and self._is_valid_matrix(matrix):
                return matrix
        return None

    def evaluate(self, predicted_answer: Any, ground_truth: Any = None, initial_state: List[List[Any]] = None) -> bool:
        """
        Evaluate predicted answer based ONLY on initial_state and Binairo game rules.
        This function validates if the predicted answer is a correct completion of the initial puzzle
        according to Binairo rules, completely ignoring the ground_truth parameter.

        Args:
            predicted_answer: The matrix extracted from the model's output or raw string
            ground_truth: IGNORED - kept for interface compatibility only
            initial_state: The initial puzzle state with None/null for empty cells

        Returns:
            True if the prediction is a valid solution to the initial_state according to Binairo rules, False otherwise
        """
        # Explicitly ignore ground_truth - this evaluation is rule-based only
        _ = ground_truth  # Explicitly mark as unused

        try:
            # Extract predicted answer if it's a string
            if isinstance(predicted_answer, str):
                extracted_matrix = self.extract_answer(predicted_answer)
                if extracted_matrix is None:
                    print(f"Failed to extract matrix from string: {predicted_answer}")
                    return False
                predicted_answer = extracted_matrix

            # Validate the matrix structure
            if not self._is_valid_matrix(predicted_answer):
                print(f"Invalid matrix structure: {predicted_answer}")
                return False

            # If no initial_state provided, validate only against Binairo rules
            if initial_state is None:
                print(f"Initial state is None: {predicted_answer}")
                return self._validate_binairo_rules(predicted_answer)

            # Normalize initial_state (handle different null representations)
            normalized_initial_state = self._normalize_initial_state(initial_state)

            # # Check if predicted answer correctly completes the initial state
            if not self._validate_completion(predicted_answer, normalized_initial_state):
                print(f"Predicted answer does not complete initial state: {predicted_answer}")
                return False

            # Check if the completed puzzle follows all Binairo rules
            return self._validate_binairo_rules(predicted_answer)

        except Exception:
            # Log the exception for debugging if needed
            return False

    def _normalize_initial_state(self, initial_state: List[List[Any]]) -> List[List[Any]]:
        """
        Normalize initial state to handle different representations of empty cells.

        Args:
            initial_state: The initial puzzle state

        Returns:
            Normalized initial state with consistent None representation for empty cells
        """
        if initial_state is None:
            return initial_state

        # If provided as a string, try to parse into a list of lists first
        if isinstance(initial_state, str):
            parsed = None
            try:
                import json
                parsed = json.loads(initial_state)
            except Exception:
                try:
                    import ast
                    parsed = ast.literal_eval(initial_state)
                except Exception:
                    parsed = None

            if isinstance(parsed, list):
                initial_state = parsed
            else:
                # Fallback: parse plain text grid with separators
                lines = [line.strip() for line in initial_state.replace(
                    '\r\n', '\n').replace('\r', '\n').split('\n') if line.strip()]
                grid: List[List[Any]] = []
                for line in lines:
                    # Split on common separators while preserving empty markers
                    tokens = [t for t in re.split(r'[\s,|]+', line)
                              if t is not None]
                    row: List[Any] = []
                    for tok in tokens:
                        tok_str = str(tok).strip()
                        if tok_str in ['0', '1']:
                            row.append(int(tok_str))
                        elif tok_str in ['', '.', '_', '-', '*', 'x', 'X']:
                            row.append(None)
                        else:
                            # Unknown token: ignore or treat as empty
                            row.append(None)
                    if len(row) > 0:
                        grid.append(row)
                initial_state = grid

        if not initial_state:
            return initial_state

        normalized = []
        for row in initial_state:
            normalized_row = []
            for cell in row:
                # Normalize different empty cell representations to None
                if (cell is None or cell == "" or cell == " "
                        or cell == "." or cell == "_"
                        or cell == 0 and isinstance(cell, str)):
                    normalized_row.append(None)
                else:
                    # Ensure numeric values are integers
                    try:
                        if isinstance(cell, str) and cell.strip() in ['0', '1']:
                            normalized_row.append(int(cell.strip()))
                        elif isinstance(cell, (int, float)) and cell in [0, 1]:
                            normalized_row.append(int(cell))
                        else:
                            normalized_row.append(None)  # Treat invalid values as empty
                    except (ValueError, TypeError):
                        normalized_row.append(None)
            normalized.append(normalized_row)

        return normalized

    def str_to_dict(self, input_str):
        if isinstance(input_str, str):
            try:
                import json
                input_str = json.loads(input_str)
            except (json.JSONDecodeError, TypeError):
                try:
                    import ast
                    input_str = ast.literal_eval(input_str)
                except (ValueError, SyntaxError):
                    pass
        return input_str

    def _validate_completion(self, predicted_answer: List[List[int]], initial_state: List[List[Any]]) -> bool:
        """
        Validate that the predicted answer correctly completes the initial state.

        Args:
            predicted_answer: The complete solution matrix
            initial_state: The initial puzzle state with None for empty cells

        Returns:
            True if predicted_answer is a valid completion of initial_state
        """
        # Ensure predicted_answer is a proper matrix (handle string inputs robustly)
        if isinstance(predicted_answer, str):
            extracted = self.extract_answer(predicted_answer)
            if extracted is not None:
                predicted_answer = extracted
            else:
                predicted_answer = self.str_to_dict(predicted_answer)

        # If still not a list of lists, or empty, fail fast
        if (not isinstance(predicted_answer, list) or len(predicted_answer) == 0
                or not isinstance(predicted_answer[0], list)):
            return False

        # Coerce string digits to integers if needed
        for i in range(len(predicted_answer)):
            row = predicted_answer[i]
            coerced_row = []
            for cell in row:
                if isinstance(cell, str):
                    cell_str = cell.strip()
                    if cell_str in ['0', '1']:
                        coerced_row.append(int(cell_str))
                    else:
                        # Leave as-is; will fail validation below
                        coerced_row.append(cell)
                else:
                    coerced_row.append(cell)
            predicted_answer[i] = coerced_row

        # Check dimensions match
        if len(predicted_answer) != len(initial_state):
            print(f"Row count mismatch: predicted {len(predicted_answer)} vs initial {len(initial_state)}")
            return False

        if len(predicted_answer) == 0:
            return False

        if len(predicted_answer[0]) != len(initial_state[0]):
            print(f"Column count mismatch: predicted {len(predicted_answer[0])} vs initial {len(initial_state[0])}")
            return False

        # Check that all pre-filled cells match
        for i in range(len(initial_state)):
            for j in range(len(initial_state[0])):
                initial_cell = initial_state[i][j]
                predicted_cell = predicted_answer[i][j]

                # If the initial cell was filled (not None), it must match the prediction
                if initial_cell is not None:
                    if initial_cell != predicted_cell:
                        print(f"Prefilled mismatch at ({i},{j}): "
                              f"initial={initial_cell}, predicted={predicted_cell}")
                        return False

                # Predicted cell must be 0 or 1
                if predicted_cell not in [0, 1]:
                    print(f"Invalid predicted cell at ({i},{j}): value={predicted_cell}")
                    return False

        return True

    def _validate_binairo_rules(self, matrix: List[List[int]]) -> bool:
        """
        Validate that the matrix follows all Binairo (Takuzu) rules.

        Rules:
        1. Each row and column must contain equal numbers of 0s and 1s
        2. No more than two consecutive identical digits in any row or column
        3. All rows must be unique
        4. All columns must be unique

        Args:
            matrix: The completed matrix to validate

        Returns:
            True if all rules are satisfied
        """
        if not matrix or len(matrix) == 0:
            return False

        # Rule 1: Equal numbers of 0s and 1s in each row and column
        if not self._check_equal_distribution(matrix):
            print(f"Equal distribution check failed: {matrix}")
            return False

        # Rule 2: No more than two consecutive identical digits
        if not self._check_no_three_consecutive(matrix):
            print(f"Three consecutive check failed: {matrix}")
            return False

        # Rule 3: All rows must be unique
        if not self._check_unique_rows(matrix):
            print(f"Unique rows check failed: {matrix}")
            return False

        # Rule 4: All columns must be unique
        if not self._check_unique_columns(matrix):
            print(f"Unique columns check failed: {matrix}")
            return False

        return True

    def _check_equal_distribution(self, matrix: List[List[int]]) -> bool:
        """Check if each row and column has equal numbers of 0s and 1s"""
        rows = len(matrix)
        cols = len(matrix[0])

        # For even-sized grids, each row/column should have equal 0s and 1s
        if rows % 2 == 0:
            expected_count = rows // 2

            # Check columns
            for col_idx in range(cols):
                column = [matrix[row_idx][col_idx] for row_idx in range(rows)]
                if column.count(0) != expected_count or column.count(1) != expected_count:
                    return False

        if cols % 2 == 0:
            expected_count = cols // 2

            # Check rows
            for row in matrix:
                if row.count(0) != expected_count or row.count(1) != expected_count:
                    return False

        return True

    def _check_no_three_consecutive(self, matrix: List[List[int]]) -> bool:
        """Check that no row or column has three consecutive identical digits"""
        rows = len(matrix)
        cols = len(matrix[0])

        # Check rows
        for row in matrix:
            for i in range(len(row) - 2):
                if row[i] == row[i + 1] == row[i + 2]:
                    return False

        # Check columns
        for col_idx in range(cols):
            column = [matrix[row_idx][col_idx] for row_idx in range(rows)]
            for i in range(len(column) - 2):
                if column[i] == column[i + 1] == column[i + 2]:
                    return False

        return True

    def _check_unique_rows(self, matrix: List[List[int]]) -> bool:
        """Check that all rows are unique"""
        row_tuples = [tuple(row) for row in matrix]
        return len(set(row_tuples)) == len(row_tuples)

    def _check_unique_columns(self, matrix: List[List[int]]) -> bool:
        """Check that all columns are unique"""
        rows = len(matrix)
        cols = len(matrix[0])

        columns = []
        for col_idx in range(cols):
            column = tuple(matrix[row_idx][col_idx] for row_idx in range(rows))
            columns.append(column)

        return len(set(columns)) == len(columns)

    def get_validation_details(self, predicted_answer: Any, initial_state: List[List[Any]] = None) -> Dict[str, Any]:
        """
        Get detailed validation results for debugging purposes.

        Args:
            predicted_answer: The matrix to validate
            initial_state: The initial puzzle state

        Returns:
            Dictionary with detailed validation results
        """
        details = {
            'valid': False,
            'errors': [],
            'extraction_successful': False,
            'matrix_well_formed': False,
            'completion_valid': False,
            'rules_satisfied': {
                'equal_distribution': False,
                'no_three_consecutive': False,
                'unique_rows': False,
                'unique_columns': False
            }
        }

        try:
            # Extract predicted answer if it's a string
            if isinstance(predicted_answer, str):
                predicted_answer = self.extract_answer(predicted_answer)

            if predicted_answer is None:
                details['errors'].append('Failed to extract matrix from input')
                return details

            details['extraction_successful'] = True

            # Validate the matrix structure
            if not self._is_valid_matrix(predicted_answer):
                details['errors'].append('Matrix is not well-formed')
                return details

            details['matrix_well_formed'] = True

            # Check completion if initial_state provided
            if initial_state is not None:
                normalized_initial = self._normalize_initial_state(initial_state)
                if not self._validate_completion(predicted_answer, normalized_initial):
                    details['errors'].append('Predicted answer does not correctly complete initial state')
                    return details
                details['completion_valid'] = True

            # Check individual rules
            details['rules_satisfied']['equal_distribution'] = self._check_equal_distribution(predicted_answer)
            details['rules_satisfied']['no_three_consecutive'] = self._check_no_three_consecutive(predicted_answer)
            details['rules_satisfied']['unique_rows'] = self._check_unique_rows(predicted_answer)
            details['rules_satisfied']['unique_columns'] = self._check_unique_columns(predicted_answer)

            # Add specific error messages for failed rules
            if not details['rules_satisfied']['equal_distribution']:
                details['errors'].append('Not all rows/columns have equal numbers of 0s and 1s')
            if not details['rules_satisfied']['no_three_consecutive']:
                details['errors'].append('Found three or more consecutive identical digits')
            if not details['rules_satisfied']['unique_rows']:
                details['errors'].append('Not all rows are unique')
            if not details['rules_satisfied']['unique_columns']:
                details['errors'].append('Not all columns are unique')

            # Overall validation
            details['valid'] = all(details['rules_satisfied'].values()) and (
                initial_state is None or details['completion_valid']
            )

        except Exception as e:
            details['errors'].append(f'Validation error: {str(e)}')

        return details
