import json
import re
from typing import Any, Dict, List, Optional, Union


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: str,
                 params: Dict[str, Any] = None) -> bool:
        raise NotImplementedError


class AquariumEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        return question

    def extract_answer(self, model_output: str) -> Optional[List[tuple]]:
        """
        Extract the model's answer from the output string with enhanced robustness.

        Supports multiple formats:
        - Answer blocks: [answer][(1,2), (3,4)][/answer]
        - Direct lists: [(1,2), (3,4)]
        - Empty lists: []
        - Scattered coordinates throughout text
        - Various bracket styles and spacing patterns
        """

        if not isinstance(model_output, str):
            return None

        # Clean and normalize the input
        normalized_output = self._normalize_text(model_output)

        # Strategy 0: Check for explicit empty list (highest priority)
        if self._is_empty_list(normalized_output):
            return []

        # Strategy 1: Find answer blocks first (highest priority)
        coordinates = self._extract_from_answer_blocks(normalized_output)
        if coordinates is not None:
            return coordinates

        # Strategy 2: Parse standard list formats
        coordinates = self._parse_list_format(normalized_output)
        if coordinates is not None:
            return coordinates

        # Strategy 3: Find scattered coordinate pairs
        coordinates = self._parse_scattered_coordinates(normalized_output)
        if coordinates is not None:
            return coordinates

        # Strategy 4: Parse comma-separated values
        coordinates = self._parse_csv_format(normalized_output)
        if coordinates is not None:
            return coordinates

        # Strategy 5: Parse structured text patterns
        coordinates = self._parse_structured_text(normalized_output)
        if coordinates is not None:
            return coordinates

        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better parsing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Normalize common variations
        text = text.replace('，', ',')  # Chinese comma
        text = text.replace('（', '(').replace('）', ')')  # Chinese parentheses
        text = text.replace('【', '[').replace('】', ']')  # Chinese brackets
        return text

    def _is_empty_list(self, text: str) -> bool:
        """Check if the text explicitly represents an empty list."""
        # Clean the text and check for empty list patterns
        _ = re.sub(r'\s+', '', text.lower())
        empty_patterns = [
            r'^\[\]$',  # Exact empty list
            r'\[\s*\]',  # Empty list with spaces
            r'answer:\s*\[\]',  # Answer: []
            r'\[answer\]\[\]\[/answer\]',  # [answer][][/answer]
        ]

        for pattern in empty_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_from_answer_blocks(self, text: str) -> Optional[List[tuple]]:
        """Extract coordinates from answer blocks like [answer]...[/answer]."""
        try:
            # Multiple patterns for answer blocks
            patterns = [
                r'\[answer\](.*?)\[/answer\]',
                r'<answer>(.*?)</answer>',
                r'Answer:\s*(.*?)(?:\n|$)',
                r'答案[：:]\s*(.*?)(?:\n|$)',
                r'Final answer:\s*(.*?)(?:\n|$)'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Use the last match (most likely the final answer)
                    last_match = matches[-1].strip()
                    coordinates = self._parse_coordinates_from_text(last_match)
                    if coordinates:
                        return coordinates

        except Exception:
            pass

        return None

    def _parse_list_format(self, text: str) -> Optional[List[tuple]]:
        """Parse coordinates in standard list format: [(1, 0), (2, 1), ...] or similar."""
        try:
            # Look for list-like structures with various bracket types
            list_patterns = [
                r'\[([^\[\]]*(?:\([^)]*\)[^\[\]]*)*)\]',  # Square brackets with parentheses inside
                r'\{([^\{\}]*(?:\([^)]*\)[^\{\}]*)*)\}',  # Curly brackets
                r'list\s*[:\=]\s*\[([^\[\]]*)\]',  # "list: [...]"
                r'coordinates?\s*[:\=]\s*\[([^\[\]]*)\]',  # "coordinates: [...]"
            ]
            if len(text) > 500:
                text = text[-100:]
            for pattern in list_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    coordinates = self._parse_coordinates_from_text(match)
                    if coordinates and len(coordinates) > 0:
                        return coordinates

        except Exception:
            pass

        return None

    def _parse_scattered_coordinates(self, text: str) -> Optional[List[tuple]]:
        """Parse coordinates scattered throughout text."""
        try:
            # Enhanced patterns for coordinate detection
            patterns = [
                r'\((\d+)\s*,\s*(\d+)\)',          # (x, y)
                r'\((\d+)\s+(\d+)\)',              # (x y)
                r'(\d+)\s*,\s*(\d+)',              # x, y
                r'cell\s*\((\d+),\s*(\d+)\)',      # cell(x, y)
                r'position\s*\((\d+),\s*(\d+)\)',  # position(x, y)
                r'(\d+)-(\d+)',                    # x-y format
            ]

            all_coordinates = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for x_str, y_str in matches:
                    try:
                        x, y = int(x_str), int(y_str)
                        # Validation with reasonable bounds
                        if 0 <= x <= 50 and 0 <= y <= 50:
                            all_coordinates.append((x, y))
                    except ValueError:
                        continue

            # Remove duplicates while preserving order
            unique_coordinates = self._remove_duplicates(all_coordinates)
            return unique_coordinates if unique_coordinates else None

        except Exception:
            pass

        return None

    def _parse_csv_format(self, text: str) -> Optional[List[tuple]]:
        """Parse coordinates in CSV-like format: x1,y1,x2,y2,... or x1 y1 x2 y2..."""
        try:
            # Extract sequences of numbers
            number_sequences = [
                re.findall(r'\d+', line) for line in text.split('\n')
                if re.search(r'\d+', line)
            ]

            for numbers in number_sequences:
                if len(numbers) >= 2 and len(numbers) % 2 == 0:
                    coordinates = []
                    for i in range(0, len(numbers), 2):
                        x, y = int(numbers[i]), int(numbers[i + 1])
                        if 0 <= x <= 50 and 0 <= y <= 50:
                            coordinates.append((x, y))

                    if coordinates:
                        return coordinates

        except Exception:
            pass

        return None

    def _parse_structured_text(self, text: str) -> Optional[List[tuple]]:
        """Parse coordinates from structured text descriptions."""
        try:
            # Look for patterns like "fill cells at (1,2), (3,4)"
            patterns = [
                r'(?:fill|water|cells?)\s+(?:at|in)?\s*[:\s]*([^\n.!?]*(?:\(\d+,\s*\d+\)[^\n.!?]*)+)',
                r'(?:solution|answer)(?:\s+is)?[:\s]*([^\n.!?]*(?:\(\d+,\s*\d+\)[^\n.!?]*)+)',
                r'(?:coordinates?|positions?)[:\s]*([^\n.!?]*(?:\(\d+,\s*\d+\)[^\n.!?]*)+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    coordinates = self._parse_coordinates_from_text(match)
                    if coordinates:
                        return coordinates

        except Exception:
            pass

        return None

    def _parse_coordinates_from_text(self, text: str) -> Optional[List[tuple]]:
        """Extract coordinate pairs from a text string."""
        try:
            # Multiple patterns to catch different coordinate formats
            coordinate_patterns = [
                r'\((\d+)\s*,\s*(\d+)\)',      # (x, y)
                r'\((\d+)\s+(\d+)\)',          # (x y)
                r'(\d+)\s*,\s*(\d+)',          # x, y
                r'(\d+)\s*-\s*(\d+)',          # x-y
            ]

            coordinates = []
            for pattern in coordinate_patterns:
                matches = re.findall(pattern, text)
                for x_str, y_str in matches:
                    try:
                        x, y = int(x_str), int(y_str)
                        if 0 <= x <= 50 and 0 <= y <= 50:
                            coordinates.append((x, y))
                    except ValueError:
                        continue

            return self._remove_duplicates(coordinates) if coordinates else None

        except Exception:
            pass

        return None

    def _remove_duplicates(self, coordinates: List[tuple]) -> List[tuple]:
        """Remove duplicates while preserving order."""
        seen = set()
        unique_coordinates = []
        for coord in coordinates:
            if coord not in seen:
                seen.add(coord)
                unique_coordinates.append(coord)
        return unique_coordinates

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Union[str, dict],
                 params: Dict[str, Any] = None) -> bool:
        """
        Evaluate if the predicted answer is correct based SOLELY on aquarium game rules and initial_state.

        Args:
            predicted_answer: Model's predicted answer (can be string or list of coordinates)
            ground_truth: Ground truth answer (IGNORED - not used in evaluation)
            initial_state: JSON string or dict containing puzzle state (regions, clues, grid_size)
            params: Additional parameters (optional)

        Returns:
            bool: True if the predicted answer satisfies all game rules, False otherwise

        Note: This function deliberately ignores ground_truth and evaluates purely based on
              game rules and initial_state to test logical consistency.
        """
        try:
            # Parse initial_state
            if isinstance(initial_state, str):
                try:
                    state_data = json.loads(initial_state)
                except json.JSONDecodeError:
                    return False
            elif isinstance(initial_state, dict):
                state_data = initial_state
            else:
                return False

            # Extract required components
            required_keys = ['regions', 'row_clues', 'col_clues', 'grid_size']
            for key in required_keys:
                if key not in state_data:
                    return False

            regions = state_data['regions']
            row_clues = state_data['row_clues']
            col_clues = state_data['col_clues']
            grid_rows, grid_cols = state_data['grid_size']

        except (KeyError, TypeError, ValueError):
            return False

        # Extract and validate predicted answer
        if isinstance(predicted_answer, str):
            extracted_answer = self.extract_answer(predicted_answer)
            if extracted_answer is None:
                return False
            predicted_coordinates = extracted_answer
        elif isinstance(predicted_answer, list):
            predicted_coordinates = predicted_answer
        else:
            return False

        # Validate coordinate format
        if not self._validate_coordinate_format(predicted_coordinates):
            return False

        # Validate coordinates are within grid bounds
        if not self._validate_coordinate_bounds(predicted_coordinates, grid_rows, grid_cols):
            return False

        # Create solution grid from predicted coordinates
        solution_grid = self._create_solution_grid(predicted_coordinates, grid_rows, grid_cols)

        # Validate all game rules
        if not self._validate_row_clues(solution_grid, row_clues, grid_rows,
                                        grid_cols):
            return False

        if not self._validate_column_clues(solution_grid, col_clues, grid_rows,
                                           grid_cols):
            return False

        if not self._validate_aquarium_rules(solution_grid, regions, grid_rows,
                                             grid_cols):
            return False

        return True

    def _validate_coordinate_format(self, coordinates: Any) -> bool:
        """Validate that coordinates is a list of tuples with two integers each."""
        if not isinstance(coordinates, list):
            return False

        if len(coordinates) == 0:
            return True  # Empty list is valid (no water)

        for i, item in enumerate(coordinates):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return False
            try:
                x, y = item
                int(x), int(y)  # Ensure they can be converted to int
            except (ValueError, TypeError):
                return False

        return True

    def _validate_coordinate_bounds(self, coordinates: List[tuple], grid_rows: int, grid_cols: int) -> bool:
        """Validate that all coordinates are within grid bounds."""
        for x, y in coordinates:
            if not (0 <= x < grid_cols and 0 <= y < grid_rows):
                return False
        return True

    def _create_solution_grid(self, coordinates: List[tuple], grid_rows: int, grid_cols: int) -> List[List[bool]]:
        """Create a 2D boolean grid from coordinate list."""
        if grid_rows == 0 or grid_cols == 0:
            return []
        solution_grid = [[False for _ in range(grid_cols)] for _ in range(grid_rows)]
        for x, y in coordinates:
            solution_grid[y][x] = True
        return solution_grid

    def _validate_row_clues(self, solution_grid: List[List[bool]], row_clues: List[int],
                            grid_rows: int, grid_cols: int) -> bool:
        """Validate that row clues match the solution."""
        if grid_rows == 0:
            return len(row_clues) == 0
        for row_idx in range(grid_rows):
            filled_count = sum(1 for col_idx in range(grid_cols) if solution_grid[row_idx][col_idx])
            expected_count = row_clues[row_idx]
            if filled_count != expected_count:
                return False
        return True

    def _validate_column_clues(self, solution_grid: List[List[bool]],
                               col_clues: List[int],
                               grid_rows: int, grid_cols: int) -> bool:
        """Validate that column clues match the solution."""
        if grid_cols == 0:
            return len(col_clues) == 0
        for col_idx in range(grid_cols):
            filled_count = sum(1 for row_idx in range(grid_rows) if solution_grid[row_idx][col_idx])
            expected_count = col_clues[col_idx]
            if filled_count != expected_count:
                return False
        return True

    def _validate_aquarium_rules(self, solution_grid: List[List[bool]],
                                 regions: List[List[int]],
                                 grid_rows: int, grid_cols: int) -> bool:
        """
        Verify aquarium game rules:
        1. Each region must be filled to a uniform water level (from bottom up)
        2. Water cannot float - if a cell is filled, the cell directly below it
           (if any, in same region) must also be filled
        """

        if grid_rows == 0 or grid_cols == 0:
            return len(regions) == 0

        # Get all unique region IDs
        unique_regions = set()
        for row in regions:
            for cell in row:
                unique_regions.add(cell)

        # For each region, verify aquarium rules
        for region_id in unique_regions:
            if not self._check_region_rules(region_id, solution_grid, regions, grid_rows, grid_cols):
                return False

        return True

    def _check_region_rules(self, region_id: int, solution_grid: List[List[bool]],
                            regions: List[List[int]], grid_rows: int,
                            grid_cols: int) -> bool:
        """Check aquarium rules for a specific region."""

        # Get all cells in this region
        region_cells = []
        for row_idx in range(grid_rows):
            for col_idx in range(grid_cols):
                if regions[row_idx][col_idx] == region_id:
                    region_cells.append((col_idx, row_idx))  # (x, y) format

        if not region_cells:
            return True  # Empty region is valid

        # Rule 1: Water cannot float - check gravity rule
        if not self._check_gravity_rule(region_cells, solution_grid, regions,
                                        region_id, grid_rows, grid_cols):
            return False

        # Rule 2: Uniform water level - group by column and check water levels
        if not self._check_uniform_water_level(region_cells, solution_grid, region_id):
            return False

        return True

    def _check_gravity_rule(self, region_cells: List[tuple],
                            solution_grid: List[List[bool]],
                            regions: List[List[int]], region_id: int,
                            grid_rows: int, grid_cols: int) -> bool:
        """Check that water cannot float: if a cell is filled, the cell directly
        below it (if any, in same region) must also be filled."""

        for x, y in region_cells:
            if solution_grid[y][x]:  # If this cell is filled
                # Check the cell directly below it
                below_y = y + 1
                if below_y < grid_rows:  # If there is a cell below
                    # Check if the cell below is in the same region
                    if regions[below_y][x] == region_id:
                        # The cell below must also be filled
                        if not solution_grid[below_y][x]:
                            return False

        return True

    def _check_uniform_water_level(self, region_cells: List[tuple],
                                   solution_grid: List[List[bool]],
                                   region_id: int) -> bool:
        """
        Check that each region has uniform water level.

        Rule: In the same region, if there's water at a certain height (y coordinate),
        then ALL cells in that region at that height should have water.
        This simulates a real aquarium where water surface is horizontal.
        """

        # Group cells by y coordinate (height level)
        levels = {}
        for x, y in region_cells:
            if y not in levels:
                levels[y] = []
            levels[y].append(x)

        # For each level, check if water is consistent
        for y_level, x_coords in levels.items():
            # Check how many cells at this level have water
            filled_count = sum(1 for x in x_coords if solution_grid[y_level][x])

            # Either all cells at this level should have water, or none should
            if 0 < filled_count < len(x_coords):
                return False

        # Also check that water levels are contiguous (from bottom up)
        # Find all levels that have water
        water_levels = []
        for y_level, x_coords in levels.items():
            if any(solution_grid[y_level][x] for x in x_coords):
                water_levels.append(y_level)

        if water_levels:
            water_levels.sort()  # Sort from top to bottom (smallest y to largest y)

            # Check that water levels are contiguous from bottom
            # Find the bottom-most water level
            bottom_water_level = max(water_levels)

            # All levels from bottom_water_level up to top water level should have water
            expected_levels = list(range(min(water_levels), bottom_water_level + 1))

            if water_levels != expected_levels:
                return False

        return True
