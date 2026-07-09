import json
import re
from ast import literal_eval
from typing import Any, Dict, List, Optional, Tuple


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: str) -> bool:
        """
        Evaluate if the predicted answer is correct.

        Args:
            predicted_answer: The model's predicted answer
            ground_truth: The ground truth answer (may not be used depending on implementation)
            initial_state: The initial state of the puzzle

        Returns:
            bool: True if the answer is correct
        """
        raise NotImplementedError


class KakuroEvaluator(BaseEvaluator):

    def extract_answer(self, model_output: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Enhanced robust extraction of answer from model output with comprehensive patterns."""
        if not model_output or not isinstance(model_output, str):
            return None

        # Clean the input text
        text = model_output.strip()

        # Strategy 1: Look for answer blocks first
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', text, re.DOTALL | re.IGNORECASE)

        # Strategy 2: Look for other common answer delimiters
        if not answer_blocks:
            delimiters = [
                r'<answer>(.*?)</answer>',
                r'answer:\s*(.*?)(?:\n\n|\n$|$)',
                r'solution:\s*(.*?)(?:\n\n|\n$|$)',
                r'final answer:\s*(.*?)(?:\n\n|\n$|$)',
                r'my answer:\s*(.*?)(?:\n\n|\n$|$)',
            ]

            for delimiter in delimiters:
                matches = re.findall(delimiter, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    answer_blocks = matches
                    break

        # Strategy 3: If no delimited blocks found, use the entire text
        if not answer_blocks:
            answer_blocks = [text]

        # Process the last (most likely) answer block
        last_block = answer_blocks[-1].strip()

        # Multiple extraction strategies for maximum robustness
        strategies = [
            self._extract_coordinate_value_patterns,
            self._extract_dict_literal,
            self._extract_structured_formats,
            self._extract_table_format,
            self._extract_loose_patterns,
            self._extract_json_like_formats,
        ]

        for strategy in strategies:
            try:
                result = strategy(last_block)
                if self._is_valid_answer_format(result):
                    return result
            except Exception:
                # Continue to next strategy if current one fails
                continue

        return None

    def _is_valid_answer_format(self, result: Any) -> bool:
        """Check if the extracted result is in valid answer format."""
        return (
            result is not None
            and isinstance(result, dict)
            and len(result) > 0
            and all(
                isinstance(k, tuple)
                and len(k) == 2
                and all(isinstance(coord, int) for coord in k) for k in result.keys()
            )
            and all(isinstance(v, int) and 1 <= v <= 9 for v in result.values())
        )

    def _extract_coordinate_value_patterns(self, text: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Extract coordinate-value patterns with comprehensive regex."""
        result = {}

        # Multiple coordinate-value patterns to handle various formats
        patterns = [
            # Standard format: (row,col):value
            r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\d+)',
            # With equals: (row,col)=value
            r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)',
            # With arrow: (row,col)->value or (row,col) -> value
            r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*->\s*(\d+)',
            # Bracket format: [row,col]:value
            r'\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*:\s*(\d+)',
            # Cell format: cell(row,col):value or Cell (row,col): value
            r'(?:cell|Cell)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[:=]\s*(\d+)',
            # Position format: pos(row,col):value
            r'(?:pos|position|Position)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[:=]\s*(\d+)',
            # Coordinate format: coord(row,col):value
            r'(?:coord|coordinate|Coordinate)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[:=]\s*(\d+)',
            # R,C format: R0C1:value or r0c1=value
            r'[Rr](\d+)[Cc](\d+)\s*[:=]\s*(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for row, col, value in matches:
                try:
                    result[(int(row), int(col))] = int(value)
                except ValueError:
                    continue

        return result if result else None

    def _extract_dict_literal(self, text: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Extract Python dictionary literal format with error handling."""
        # Try to find dictionary-like structures
        dict_patterns = [
            r'\{[^}]*\}',  # Basic dictionary pattern
            r'{\s*[\s\S]*?\s*}',  # Multi-line dictionary
        ]

        for pattern in dict_patterns:
            dict_matches = re.findall(pattern, text, re.DOTALL)

            for dict_str in dict_matches:
                try:
                    # Try direct evaluation
                    answer_dict = literal_eval(dict_str)
                    if isinstance(answer_dict, dict):
                        result = {}
                        for k, v in answer_dict.items():
                            if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
                                # Handle "(0,1)" format string keys
                                coord_str = k.strip('()')
                                row, col = map(int, coord_str.split(','))
                                result[(row, col)] = int(v)
                            elif isinstance(k, tuple) and len(k) == 2:
                                result[k] = int(v)
                            elif isinstance(k, str) and ',' in k:
                                # Handle "0,1" format
                                row, col = map(int, k.split(','))
                                result[(row, col)] = int(v)

                        if result:
                            return result

                except (ValueError, SyntaxError):
                    # Try manual parsing for malformed dictionaries
                    result = self._parse_malformed_dict(dict_str)
                    if result:
                        return result

        return None

    def _parse_malformed_dict(self, dict_str: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Parse malformed dictionary strings manually."""
        result = {}

        # Remove outer braces and split by commas
        content = dict_str.strip('{}')

        # Split by commas, but be careful about commas within coordinates
        items = re.split(r',(?=\s*["\'\(])', content)

        for item in items:
            item = item.strip()
            # Look for key-value pairs
            kv_patterns = [
                r'["\']?\(\s*(\d+)\s*,\s*(\d+)\s*\)["\']?\s*:\s*(\d+)',
                r'["\']?(\d+)\s*,\s*(\d+)["\']?\s*:\s*(\d+)',
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\d+)',
            ]

            for pattern in kv_patterns:
                match = re.search(pattern, item)
                if match:
                    row, col, value = match.groups()
                    try:
                        result[(int(row), int(col))] = int(value)
                        break
                    except ValueError:
                        continue

        return result if result else None

    def _extract_structured_formats(self, text: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Extract structured formats like lists, comma-separated, newline-separated."""
        result = {}

        # Format 1: List-like formats
        list_patterns = [
            r'\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)\s*\]',  # [(0,1), 3]
            r'\(\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)\s*\)',  # ((0,1), 3)
            r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',  # [0, 1, 3]
        ]

        for pattern in list_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    try:
                        row, col, value = map(int, match)
                        result[(row, col)] = value
                    except ValueError:
                        continue

        # Format 2: Space-separated entries
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # Try to extract from each line
                coord_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[:=]\s*(\d+)', line)
                for row, col, value in coord_matches:
                    try:
                        result[(int(row), int(col))] = int(value)
                    except ValueError:
                        continue

        return result if result else None

    def _extract_table_format(self, text: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Extract table-like formats."""
        result = {}
        lines = text.split('\n')

        # Look for table-like structures
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('|', '-', '+')):
                continue

            # Table row format: | (0,1) | 3 | or similar
            table_patterns = [
                r'\|\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\|\s*(\d+)\s*\|',
                r'(\d+)\s*,\s*(\d+)\s*\|\s*(\d+)',
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\|\s*(\d+)',
            ]

            for pattern in table_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if len(match) == 3:
                        try:
                            row, col, value = map(int, match)
                            result[(row, col)] = value
                        except ValueError:
                            continue

        return result if result else None

    def _extract_loose_patterns(self, text: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Loose pattern matching for edge cases and unusual formats."""
        result = {}

        # Very flexible patterns
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Pattern 1: Any format with "row" and "col" keywords
            row_col_patterns = [
                r'(?:row|r)\s*[:=]?\s*(\d+).*?(?:col|column|c)\s*[:=]?\s*(\d+).*?(?:value|val|v)\s*[:=]?\s*(\d+)',
                r'(?:col|column|c)\s*[:=]?\s*(\d+).*?(?:row|r)\s*[:=]?\s*(\d+).*?(?:value|val|v)\s*[:=]?\s*(\d+)',
            ]

            for pattern in row_col_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        if pattern.startswith('(?:row'):
                            row, col, value = map(int, match)
                        else:
                            col, row, value = map(int, match)
                        result[(row, col)] = value
                    except ValueError:
                        continue

            # Pattern 2: Simple number sequences that might represent coordinates and values
            # Format: "row col value" on separate lines or separated by spaces/commas
            number_sequences = re.findall(r'\b(\d+)\s*[,\s]\s*(\d+)\s*[,\s:=]\s*(\d+)\b', line)
            for row, col, value in number_sequences:
                try:
                    r, c, v = int(row), int(col), int(value)
                    # Basic sanity check: coordinates shouldn't be too large and value should be 1-9
                    if 0 <= r <= 20 and 0 <= c <= 20 and 1 <= v <= 9:
                        result[(r, c)] = v
                except ValueError:
                    continue

        return result if result else None

    def _extract_json_like_formats(self, text: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Extract JSON-like formats that might not be valid Python."""
        result = {}

        # Find JSON-like structures
        json_patterns = [
            r'\{[^}]*\}',
            r'{\s*[\s\S]*?\s*}',
        ]

        for pattern in json_patterns:
            json_matches = re.findall(pattern, text, re.DOTALL)

            for json_str in json_matches:
                try:
                    # Try JSON parsing first
                    import json as json_module
                    data = json_module.loads(json_str)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(k, str):
                                # Handle various key formats
                                if k.startswith('(') and k.endswith(')'):
                                    coord_str = k.strip('()')
                                    row, col = map(int, coord_str.split(','))
                                    result[(row, col)] = int(v)
                                elif ',' in k:
                                    row, col = map(int, k.split(','))
                                    result[(row, col)] = int(v)

                        if result:
                            return result
                except (json.JSONDecodeError, ValueError):
                    continue

        return result if result else None

    def parse_solution_string(self, solution_str: str) -> Optional[Dict[Tuple[int, int], int]]:
        """Parse a solution string using the enhanced extraction methods."""
        return self.extract_answer(solution_str)

    def validate_kakuro_solution(self, predicted_answer: Dict[Tuple[int, int], int], initial_state: str) -> bool:
        """
        Validate a Kakuro solution based only on the initial state and game rules.

        Args:
            predicted_answer: Dictionary mapping (row, col) tuples to digit values
            initial_state: JSON string representing the grid structure

        Returns:
            bool: True if the solution is valid, False otherwise
        """
        # Support both JSON string and already-parsed Python structures
        try:
            if isinstance(initial_state, str):
                grid = json.loads(initial_state)
            else:
                grid = initial_state  # assume it's already a parsed list[list[dict]]
        except (json.JSONDecodeError, TypeError):
            return False

        if not isinstance(predicted_answer, dict):
            return False

        if not predicted_answer:  # Empty answer
            return False

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # 1. Check that all values are digits 1-9
        for coord, value in predicted_answer.items():
            if not isinstance(value, int) or value < 1 or value > 9:
                return False

            # Check that coordinate is valid
            if not isinstance(coord, tuple) or len(coord) != 2:
                return False

            row, col = coord
            if row < 0 or row >= rows or col < 0 or col >= cols:
                return False

            # Check that the cell is actually a white cell
            if grid[row][col]['type'] != 'white':
                return False

        # 2. Check that all white cells are filled
        white_cells = set()
        for i in range(rows):
            for j in range(cols):
                if grid[i][j]['type'] == 'white':
                    white_cells.add((i, j))

        if set(predicted_answer.keys()) != white_cells:
            return False

        # 3. Validate all constraint runs
        for i in range(rows):
            for j in range(cols):
                cell = grid[i][j]
                if cell['type'] == 'black':
                    # Check right constraint
                    if 'right' in cell:
                        right_hint = cell['right']
                        # Determine target sum and run length policy
                        target_sum: Optional[int] = None
                        expected_length: Optional[int] = None
                        if isinstance(right_hint, (list, tuple)) and len(right_hint) == 2:
                            try:
                                target_sum = int(right_hint[0])
                                expected_length = int(right_hint[1])
                            except (ValueError, TypeError):
                                return False
                        else:
                            try:
                                target_sum = int(right_hint)
                            except (ValueError, TypeError):
                                return False

                        # Collect cells in the right run until a non-white cell or boundary
                        run_cells: List[Tuple[int, int]] = []
                        k = j + 1
                        while k < cols and grid[i][k]['type'] == 'white':
                            run_cells.append((i, k))
                            k += 1

                        # If length was provided, enforce it; otherwise infer from layout
                        if expected_length is not None and len(run_cells) != expected_length:
                            return False

                        # Check all cells are in prediction
                        if not all(rc in predicted_answer for rc in run_cells):
                            return False

                        # Check sum
                        actual_sum = sum(predicted_answer[rc] for rc in run_cells)
                        if actual_sum != target_sum:
                            return False

                        # Check uniqueness (no repeated digits)
                        values = [predicted_answer[rc] for rc in run_cells]
                        if len(set(values)) != len(values):
                            return False

                    # Check down constraint
                    if 'down' in cell:
                        down_hint = cell['down']
                        target_sum: Optional[int] = None
                        expected_length: Optional[int] = None
                        if isinstance(down_hint, (list, tuple)) and len(down_hint) == 2:
                            try:
                                target_sum = int(down_hint[0])
                                expected_length = int(down_hint[1])
                            except (ValueError, TypeError):
                                return False
                        else:
                            try:
                                target_sum = int(down_hint)
                            except (ValueError, TypeError):
                                return False

                        # Collect cells in the down run until a non-white cell or boundary
                        run_cells: List[Tuple[int, int]] = []
                        k = i + 1
                        while k < rows and grid[k][j]['type'] == 'white':
                            run_cells.append((k, j))
                            k += 1

                        # If length was provided, enforce it; otherwise infer from layout
                        if expected_length is not None and len(run_cells) != expected_length:
                            return False

                        # Check all cells are in prediction
                        if not all(rc in predicted_answer for rc in run_cells):
                            return False

                        # Check sum
                        actual_sum = sum(predicted_answer[rc] for rc in run_cells)
                        if actual_sum != target_sum:
                            return False

                        # Check uniqueness (no repeated digits)
                        values = [predicted_answer[rc] for rc in run_cells]
                        if len(set(values)) != len(values):
                            return False

        return True

    def _normalize_answer(self, ans: Any) -> Optional[Dict[Tuple[int, int], int]]:
        """Convert various answer formats to {(row, col): val} dict; return None if invalid/empty."""
        if ans is None:
            return None

        # 字符串：用已有的 extract_answer 做鲁棒解析
        if isinstance(ans, str):
            ans = self.extract_answer(ans)
        # 依然可能是字符串解析失败，或已经是 dict
        if not isinstance(ans, dict) or not ans:
            return None

        # 统一把字符串坐标转成 tuple
        normalized: Dict[Tuple[int, int], int] = {}
        for k, v in ans.items():
            try:
                if isinstance(k, tuple) and len(k) == 2:
                    r, c = int(k[0]), int(k[1])
                elif isinstance(k, str) and k.startswith('(') and k.endswith(')'):
                    r, c = map(int, k.strip('()').split(','))
                elif isinstance(k, str) and ',' in k:
                    r, c = map(int, k.split(','))
                else:
                    return None
                v = int(v)
                if not (1 <= v <= 9):
                    return None
                normalized[(r, c)] = v
            except Exception:
                return None
        return normalized if normalized else None

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: str) -> bool:
        """
        如果提供了 ground_truth，则优先做“答案对答案”的精确匹配；
        否则才进行基于初始盘面规则的完整解校验。
        """
        # 1) 若同时提供了 GT：做精确匹配（支持单格/部分答案评测）
        gt_norm = self._normalize_answer(ground_truth) if ground_truth is not None else None
        if gt_norm is not None:
            pred_norm = self._normalize_answer(predicted_answer)
            return pred_norm is not None and pred_norm == gt_norm

        # 2) 无 GT：按规则校验完整解
        pred_norm = self._normalize_answer(predicted_answer)
        if pred_norm is None:
            return False
        return self.validate_kakuro_solution(pred_norm, initial_state)

    # def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: str) -> bool:
    #     """
    #     Evaluate if the predicted answer is correct based only on initial state and game rules.
    #     Note: ground_truth is ignored in this implementation as evaluation is rule-based.

    #     Args:
    #         predicted_answer: The model's predicted answer (dict, string, or raw model output)
    #         ground_truth: The ground truth answer (ignored in this implementation)
    #         initial_state: JSON string representing the grid structure

    #     Returns:
    #         bool: True if the solution satisfies all game constraints
    #     """
    #     if predicted_answer is None:
    #         return False

    #     # If it's a string, try to extract the answer using robust parsing
    #     if isinstance(predicted_answer, str):
    #         predicted_answer = self.extract_answer(predicted_answer)

    #     if predicted_answer is None:
    #         return False

    #     if not isinstance(predicted_answer, dict):
    #         return False

    #     # Convert string coordinates to tuples if needed
    #     if predicted_answer and all(isinstance(k, str) for k in predicted_answer.keys()):
    #         converted_answer = {}
    #         for k, v in predicted_answer.items():
    #             try:
    #                 if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
    #                     coord_str = k.strip('()')
    #                     row, col = map(int, coord_str.split(','))
    #                     converted_answer[(row, col)] = int(v)
    #                 elif isinstance(k, str) and ',' in k:
    #                     row, col = map(int, k.split(','))
    #                     converted_answer[(row, col)] = int(v)
    #                 else:
    #                     return False
    #             except (ValueError, AttributeError):
    #                 return False
    #         predicted_answer = converted_answer

    #     # Validate the solution using only initial_state and game rules
    #     return self.validate_kakuro_solution(predicted_answer, initial_state)
