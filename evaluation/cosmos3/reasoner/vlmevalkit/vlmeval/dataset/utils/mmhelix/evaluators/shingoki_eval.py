import json
import re
from typing import Any, Dict, List, Tuple, Union


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class ShingokiEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        puzzle_image = params.get('image', '')
        rows = params.get('rows', 0)
        cols = params.get('cols', 0)

        prompt = [
            "# Shingoki Puzzle",
            "",
            "## Rules",
            "- Draw a single continuous loop without crossings or branches",
            "- White circles must be passed through in a straight line",
            "- Black circles must be turned upon",
            "- Numbers in circles show the sum of the lengths of the 2 straight lines going out of that circle",
            "",
            f"## Puzzle Grid ({rows}x{cols})",
            "",
            f"![Shingoki Puzzle]({puzzle_image})",
            "",
            "## Instructions",
            "1. Analyze the image and identify all circles and their values",
            "2. Solve the puzzle step-by-step",
            "3. Provide your answer as a list of connected line segments that form the loop",
            "",
            "## Answer Format",
            "Your answer should be in the following format:",
            "```",
            "(r1,c1)-(r2,c2) (r2,c2)-(r3,c3) ...",
            "```",
            "Where each (r,c) represents the row and column coordinates of a grid point.",
        ]

        return "\n".join(prompt)

    def extract_answer(self, model_output: str) -> Union[List[Tuple[Tuple[int, int], Tuple[int, int]]], None]:
        """
        Extract the answer from the model's output with enhanced robustness

        Args:
            model_output: The text output from the model

        Returns:
            A list of line segments or None if no valid answer found
        """
        if not model_output or not isinstance(model_output, str):
            return None

        # Clean the input: remove extra whitespace and normalize
        cleaned_output = re.sub(r'\s+', ' ', model_output.strip())

        # Strategy 1: Look for standard format (r1,c1)-(r2,c2)
        pattern1 = r'\((\d+),(\d+)\)-\((\d+),(\d+)\)'
        matches1 = re.findall(pattern1, cleaned_output)

        if matches1:
            segments = []
            for match in matches1:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 2: Look for format with spaces around coordinates (r1, c1) - (r2, c2)
        pattern2 = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[-–—]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches2 = re.findall(pattern2, cleaned_output)

        if matches2:
            segments = []
            for match in matches2:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 3: Look for format without parentheses: r1,c1-r2,c2
        pattern3 = r'(\d+),(\d+)[-–—](\d+),(\d+)'
        matches3 = re.findall(pattern3, cleaned_output)

        if matches3:
            segments = []
            for match in matches3:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 4: Look for format with square brackets [r1,c1]-[r2,c2]
        pattern4 = r'\[(\d+),(\d+)\][-–—]\[(\d+),(\d+)\]'
        matches4 = re.findall(pattern4, cleaned_output)

        if matches4:
            segments = []
            for match in matches4:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 5: Look for coordinates separated by various delimiters
        # Pattern like: 1,0 to 0,0 or 1,0 -> 0,0 or 1,0 → 0,0
        pattern5 = r'(\d+),(\d+)\s*(?:[-–—]|to|->|→|=>)\s*(\d+),(\d+)'
        matches5 = re.findall(pattern5, cleaned_output)

        if matches5:
            segments = []
            for match in matches5:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 6: Look for format with curly braces {r1,c1}-{r2,c2}
        pattern6 = r'\{(\d+),(\d+)\}[-–—]\{(\d+),(\d+)\}'
        matches6 = re.findall(pattern6, cleaned_output)

        if matches6:
            segments = []
            for match in matches6:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 7: Look for coordinates with different separators (semicolon, pipe, etc.)
        pattern7 = r'(\d+)[,;|:](\d+)\s*[-–—]\s*(\d+)[,;|:](\d+)'
        matches7 = re.findall(pattern7, cleaned_output)

        if matches7:
            segments = []
            for match in matches7:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 8: Look for line breaks or comma-separated segments
        separators = ['\n', ';', ',']
        for sep in separators:
            parts = cleaned_output.split(sep)
            segments = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Try to extract segment from this part using any of the above patterns
                for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7]:
                    match = re.search(pattern, part)
                    if match:
                        try:
                            r1, c1, r2, c2 = map(int, match.groups())
                            segments.append(((r1, c1), (r2, c2)))
                            break
                        except ValueError:
                            continue

            if segments:
                return segments

        # Strategy 9: Look for JSON-like format
        try:
            # Try to find JSON-like structures
            json_pattern = r'\{[^}]*"answer"[^}]*\}'
            json_matches = re.findall(json_pattern, cleaned_output)

            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if "answer" in data:
                        answer_data = data["answer"]
                        if isinstance(answer_data, str):
                            return self.extract_answer(answer_data)
                        elif isinstance(answer_data, list):
                            # Try to parse list as segments
                            segments = []
                            for item in answer_data:
                                if isinstance(item, str):
                                    sub_segments = self.extract_answer(item)
                                    if sub_segments:
                                        segments.extend(sub_segments)
                            if segments:
                                return segments
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        except Exception:
            pass

        # Strategy 10: Look for sequences of coordinates that might represent a path
        # Find all coordinate pairs in the text
        coord_pattern = r'\(?(\d+)[,;:\s]+(\d+)\)?'
        coord_matches = re.findall(coord_pattern, cleaned_output)

        if len(coord_matches) >= 2:
            # Try to form segments by connecting consecutive coordinates
            segments = []
            coords = []
            for match in coord_matches:
                try:
                    r, c = map(int, match)
                    coords.append((r, c))
                except ValueError:
                    continue

            # Connect consecutive coordinates
            for i in range(len(coords) - 1):
                segments.append((coords[i], coords[i + 1]))

            if segments:
                return segments

        # Strategy 11: Look for coordinates listed with explicit "from" and "to" keywords
        pattern11 = r'from\s*\(?(\d+)[,\s]+(\d+)\)?\s*to\s*\(?(\d+)[,\s]+(\d+)\)?'
        matches11 = re.findall(pattern11, cleaned_output, re.IGNORECASE)

        if matches11:
            segments = []
            for match in matches11:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 12: Look for numbered coordinates with patterns like "1. (r1,c1) to (r2,c2)"
        pattern12 = r'\d+\.\s*\(?(\d+)[,\s]+(\d+)\)?\s*(?:to|->|→)\s*\(?(\d+)[,\s]+(\d+)\)?'
        matches12 = re.findall(pattern12, cleaned_output)

        if matches12:
            segments = []
            for match in matches12:
                try:
                    r1, c1, r2, c2 = map(int, match)
                    segments.append(((r1, c1), (r2, c2)))
                except ValueError:
                    continue
            if segments:
                return segments

        # Strategy 13: Extract all numbers and try to group them into segments
        # This is a last resort strategy
        numbers = re.findall(r'\d+', cleaned_output)

        if len(numbers) >= 4 and len(numbers) % 4 == 0:
            # Try to group numbers into coordinate pairs
            segments = []
            for i in range(0, len(numbers), 4):
                try:
                    r1, c1, r2, c2 = map(int, numbers[i:i + 4])
                    segments.append(((r1, c1), (r2, c2)))
                except (ValueError, IndexError):
                    break

            if segments:
                return segments

        return None

    def evaluate(self, predicted_answer: Any, ground_truth: Any = None, initial_state: Any = None) -> bool:
        """
        Evaluate if the predicted answer is correct based solely on initial state and game rules

        Args:
            predicted_answer: The predicted answer (can be string or parsed segments)
            ground_truth: The ground truth answer (not used for validation, kept for API compatibility)
            initial_state: The initial state of the puzzle containing grid and circle information

        Returns:
            True if the answer is correct according to game rules, False otherwise
        """
        # Initialize debug information for detailed error tracking
        debug_info = {"errors": [], "warnings": []}

        try:
            # Step 1: Extract and validate predicted_answer
            if predicted_answer is None:
                return False

            # Extract predicted_answer using extract_answer if it's a string
            if isinstance(predicted_answer, str):
                predicted_segments = self.extract_answer(predicted_answer)
                if predicted_segments is None:
                    return False
                if not predicted_segments:
                    return False
            elif isinstance(predicted_answer, list):
                predicted_segments = predicted_answer
                if not predicted_segments:
                    return False
            else:
                return False

            # Step 2: Validate initial_state
            if initial_state is None:
                return False

            # Parse and normalize initial_state
            try:
                # Decode JSON string if needed
                if isinstance(initial_state, str):
                    state_data = json.loads(initial_state)
                else:
                    state_data = initial_state

                # Normalize into expected dict form with a 'grid' key
                if isinstance(state_data, list):
                    # Dataset provides a raw 2D grid as JSON array
                    state_data = {"grid": state_data}
                elif isinstance(state_data, dict):
                    # Use as-is; caller may provide {'grid': ...} or {'rows': ..., 'cols': ..., 'circles': ...}
                    pass
                else:
                    return False
            except json.JSONDecodeError:
                return False

            # Step 3: Validate segments format
            if not self._validate_segments_format(predicted_segments, debug_info):
                return False

            # Step 4: Validate using initial state and game rules
            validation_result = self._validate_with_initial_state(predicted_segments, state_data, debug_info)

            return validation_result

        except Exception:
            return False

    def _validate_segments_format(self, segments: List, debug_info: Dict) -> bool:
        """
        Validate that segments have the correct format

        Args:
            segments: List of segments to validate
            debug_info: Dictionary to store debugging information

        Returns:
            True if format is valid, False otherwise
        """
        if not isinstance(segments, list):
            debug_info["errors"].append("Segments is not a list")
            return False

        for i, segment in enumerate(segments):
            if not isinstance(segment, tuple) or len(segment) != 2:
                debug_info["errors"].append(f"Segment {i} is not a tuple of length 2: {segment}")
                return False

            p1, p2 = segment

            if not isinstance(p1, tuple) or len(p1) != 2:
                debug_info["errors"].append(f"Segment {i} first point is not a tuple of length 2: {p1}")
                return False

            if not isinstance(p2, tuple) or len(p2) != 2:
                debug_info["errors"].append(f"Segment {i} second point is not a tuple of length 2: {p2}")
                return False

            # Validate that coordinates are integers
            try:
                r1, c1 = p1
                r2, c2 = p2
                int(r1), int(c1), int(r2), int(c2)
            except (ValueError, TypeError):
                debug_info["errors"].append(f"Segment {i} contains non-integer coordinates: {segment}")
                return False

        return True

    def _validate_with_initial_state(self, segments: List[Tuple], state_data: Dict, debug_info: Dict) -> bool:
        """
        Validate the solution against the initial state and game rules

        Args:
            segments: List of line segments forming the solution
            state_data: Initial state data containing grid and circles information
            debug_info: Dictionary to store debugging information

        Returns:
            True if solution is valid according to game rules, False otherwise
        """
        # Extract grid information
        if 'grid' in state_data:
            grid = state_data['grid']
            rows = len(grid)
            cols = len(grid[0]) if grid else 0
        else:
            rows = state_data.get('rows', 0)
            cols = state_data.get('cols', 0)

        if rows == 0 or cols == 0:
            debug_info["errors"].append(f"Invalid grid dimensions: {rows}x{cols}")
            return False

        # Extract circles information
        circles = {}
        if 'circles' in state_data:
            circles = state_data['circles']
        elif 'grid' in state_data:
            # Parse circles from grid representation
            circles = self._parse_circles_from_grid(state_data['grid'])

        if not circles:
            debug_info["errors"].append("No circles found in the puzzle")
            return False

        # Rule 1: All segments must be within grid boundaries and adjacent
        if not self._validate_segments_boundaries(segments, rows, cols):
            debug_info["errors"].append("Segments violate grid boundaries or adjacency rules")
            return False

        # Rule 2: Must form a single continuous loop
        if not self._validate_single_continuous_loop(segments):
            debug_info["errors"].append("Path does not form a single continuous loop")
            return False

        # Rule 3: Must pass through ALL circles
        if not self._validate_path_through_circles(segments, circles):
            debug_info["errors"].append("Path does not pass through all circles")
            return False

        # Rule 4: Validate circle constraints (white/black circle rules and values)
        circle_validation_result = self._validate_all_circle_constraints(segments, circles, debug_info)
        if not circle_validation_result:
            # Detailed errors are already added to debug_info in _validate_all_circle_constraints
            return False

        return True

    def _parse_circles_from_grid(self, grid: List[List[str]]) -> Dict:
        """
        Parse circles from grid representation

        Args:
            grid: 2D grid with circle representations

        Returns:
            Dictionary of circles with position keys and type/value info
        """
        circles = {}

        for r in range(len(grid)):
            for c in range(len(grid[r])):
                cell = grid[r][c]
                if cell and cell != '.':
                    # Parse circle type and value from cell content
                    if cell.startswith('W'):  # White circle
                        try:
                            value = int(cell[1:])
                            pos_key = f"{r},{c}"
                            circles[pos_key] = {"type": "white", "value": value}
                        except ValueError:
                            continue
                    elif cell.startswith('B'):  # Black circle
                        try:
                            value = int(cell[1:])
                            pos_key = f"{r},{c}"
                            circles[pos_key] = {"type": "black", "value": value}
                        except ValueError:
                            continue

        return circles

    def _validate_single_continuous_loop(self, segments: List[Tuple]) -> bool:
        """
        Validate that the segments form a single continuous loop

        Args:
            segments: List of line segments

        Returns:
            True if segments form a valid single continuous loop
        """
        if not segments:
            return False

        # Build adjacency graph
        graph = {}
        for p1, p2 in segments:
            if p1 not in graph:
                graph[p1] = []
            if p2 not in graph:
                graph[p2] = []
            graph[p1].append(p2)
            graph[p2].append(p1)

        # Rule: Each point must have exactly 2 connections (loop property)
        for point, neighbors in graph.items():
            if len(neighbors) != 2:
                return False

        # Rule: Must form exactly one connected component
        if not graph:
            return False

        visited = set()
        start_point = next(iter(graph.keys()))

        def dfs(point):
            visited.add(point)
            for neighbor in graph[point]:
                if neighbor not in visited:
                    dfs(neighbor)

        dfs(start_point)

        # All points should be visited (single connected component)
        return len(visited) == len(graph)

    def _validate_segments_boundaries(self, segments: List[Tuple], rows: int, cols: int) -> bool:
        """
        Validate that all segments are within boundaries and connect adjacent points

        Args:
            segments: List of line segments
            rows: Number of rows in grid (grid cells)
            cols: Number of columns in grid (grid cells)

        Returns:
            True if all segments are valid
        """
        for segment in segments:
            p1, p2 = segment
            r1, c1 = p1
            r2, c2 = p2

            # Check boundaries - for an NxM grid of cells, we have (N+1)x(M+1) grid points
            # So coordinates should be from 0 to N and 0 to M (inclusive)
            if not (0 <= r1 <= rows and 0 <= c1 <= cols
                    and 0 <= r2 <= rows and 0 <= c2 <= cols):
                return False

            # Check adjacency (Manhattan distance = 1, no diagonal connections)
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                return False

        return True

    def _validate_path_through_circles(self, segments: List[Tuple], circles: Dict) -> bool:
        """
        Validate that the path passes through all circles

        Args:
            segments: List of line segments
            circles: Dictionary of circles

        Returns:
            True if path passes through all circles
        """
        # Build set of all points on the path
        path_points = set()
        for p1, p2 in segments:
            path_points.add(p1)
            path_points.add(p2)

        # Convert circle positions to tuples
        circle_positions = set()
        for pos_str in circles.keys():
            r, c = map(int, pos_str.split(','))
            circle_positions.add((r, c))

        # Rule: Path must pass through ALL circles
        return circle_positions.issubset(path_points)

    def _validate_all_circle_constraints(self, segments: List[Tuple], circles: Dict, debug_info: Dict) -> bool:
        """
        Validate all circle constraints (white/black rules and values)

        Args:
            segments: List of line segments
            circles: Dictionary of circles
            debug_info: Dictionary to store debugging information

        Returns:
            True if all circle constraints are satisfied
        """
        # Build adjacency graph
        graph = {}
        for p1, p2 in segments:
            if p1 not in graph:
                graph[p1] = []
            if p2 not in graph:
                graph[p2] = []
            graph[p1].append(p2)
            graph[p2].append(p1)

        # Check each circle constraint
        for pos_str, circle_info in circles.items():
            # Convert position string to tuple
            r, c = map(int, pos_str.split(','))
            point = (r, c)

            # Circle must be on the path
            if point not in graph:
                debug_info["errors"].append(f"Circle at {point} is not on the path")
                return False

            # Circle must have exactly 2 connections
            if len(graph[point]) != 2:
                debug_info["errors"].append(f"Circle at {point} does not have exactly 2 connections")
                return False

            circle_type = circle_info["type"]
            expected_value = circle_info["value"]

            neighbors = graph[point]

            # Validate white circle constraint (straight line)
            if circle_type == "white":
                if not self._validate_white_circle_constraint(point, neighbors):
                    debug_info["errors"].append(f"White circle at {point} is not on a straight line")
                    return False

            # Validate black circle constraint (turn)
            elif circle_type == "black":
                if not self._validate_black_circle_constraint(point, neighbors):
                    debug_info["errors"].append(f"Black circle at {point} is not at a turning point")
                    return False

            # Validate circle value (sum of line lengths)
            actual_value = self._calculate_circle_value(point, neighbors, graph)
            if actual_value != expected_value:
                debug_info["errors"].append(
                    f"Circle at {point} has value {expected_value} but actual length is {actual_value}")
                return False

        return True

    def _validate_white_circle_constraint(self, point: Tuple[int, int], neighbors: List[Tuple[int, int]]) -> bool:
        """
        Validate that a white circle is on a straight line

        Args:
            point: Circle position
            neighbors: Two neighboring points

        Returns:
            True if the circle is on a straight line
        """
        if len(neighbors) != 2:
            return False

        n1, n2 = neighbors
        r, c = point

        # Calculate directions from point to neighbors
        dir1 = (n1[0] - r, n1[1] - c)
        dir2 = (n2[0] - r, n2[1] - c)

        # For straight line, directions should be opposite
        return (dir1[0] + dir2[0] == 0) and (dir1[1] + dir2[1] == 0)

    def _validate_black_circle_constraint(self, point: Tuple[int, int], neighbors: List[Tuple[int, int]]) -> bool:
        """
        Validate that a black circle is at a turning point

        Args:
            point: Circle position
            neighbors: Two neighboring points

        Returns:
            True if the circle is at a turning point
        """
        if len(neighbors) != 2:
            return False

        n1, n2 = neighbors
        r, c = point

        # Calculate directions from point to neighbors
        dir1 = (n1[0] - r, n1[1] - c)
        dir2 = (n2[0] - r, n2[1] - c)

        # For turning point, directions should NOT be opposite
        return not ((dir1[0] + dir2[0] == 0) and (dir1[1] + dir2[1] == 0))

    def _calculate_circle_value(self, circle_point: Tuple[int, int],
                                neighbors: List[Tuple[int, int]],
                                graph: Dict) -> int:
        """
        Calculate the actual value for a circle (sum of straight line lengths in both directions)

        Args:
            circle_point: Circle position
            neighbors: Two neighboring points
            graph: Adjacency graph of the entire path

        Returns:
            Sum of the two straight line segment lengths from the circle
        """
        if len(neighbors) != 2:
            return 0

        total_length = 0

        # Calculate length in both directions from the circle
        for neighbor in neighbors:
            length = self._get_straight_line_length_from_circle(circle_point, neighbor, graph)
            total_length += length

        return total_length

    def _get_straight_line_length_from_circle(
            self, circle_point: Tuple[int, int],
            next_point: Tuple[int, int],
            graph: Dict) -> int:
        """
        Get the length of a straight line segment starting from circle going towards next_point

        Args:
            circle_point: Starting circle point
            next_point: Next point in the direction
            graph: Adjacency graph

        Returns:
            Length of the straight line segment from the circle
        """
        # Direction vector from circle to next
        direction = (next_point[0] - circle_point[0], next_point[1] - circle_point[1])

        current = next_point
        length = 1  # Count the first segment from circle to next_point
        visited = {circle_point}  # Avoid going back to the circle

        while True:
            visited.add(current)

            # Find the next point that continues in the same direction
            next_in_direction = None

            for neighbor in graph[current]:
                if neighbor not in visited:
                    # Check if this neighbor is in the same direction
                    current_to_neighbor = (neighbor[0] - current[0], neighbor[1] - current[1])
                    if current_to_neighbor == direction:
                        next_in_direction = neighbor
                        break

            if next_in_direction is None:
                # No more points in this direction, stop
                break

            # Continue in the same direction
            current = next_in_direction
            length += 1

        return length

    # Keep existing methods for backward compatibility
    def validate_loop(self, segments: List[Tuple], rows: int = 0, cols: int = 0) -> bool:
        """Backward compatibility wrapper"""
        return self._validate_single_continuous_loop(segments)

    def validate_circle_constraints(self, segments: List[Tuple], circles: Dict, rows: int, cols: int) -> bool:
        """Backward compatibility wrapper"""
        debug_info = {"errors": []}
        return self._validate_all_circle_constraints(segments, circles, debug_info)
