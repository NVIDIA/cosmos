import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


class BaseEvaluator:

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any,
                 initial_state: Any) -> bool:
        raise NotImplementedError


class BridgesEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """
        Prepare the prompt for the bridges task.
        Returns the question with additional instructions if needed.
        """
        return question

    def extract_answer(self, model_output: str) -> str:
        """
        Enhanced extraction of bridge specifications from model output.
        Handles various formats and edge cases with robust regex patterns.
        """
        if not model_output or not isinstance(model_output, str):
            return ""

        # Clean the input first
        model_output = model_output.strip()

        # Multiple patterns to handle different answer formats
        patterns = [
            # Standard format: <answer>content</answer>
            r'<answer>\s*(.*?)\s*</answer>',
            # Alternative format: [answer]content[/answer]
            r'\[answer\]\s*(.*?)\s*\[/answer\]',
            # JSON-like format: "answer": "content" (handle escaped newlines)
            r'"answer"\s*:\s*"(.*?)"',
            # Direct format without tags (Answer: or Solution:)
            r'(?:Answer|Solution|答案)[:\s]*\n?(.*?)(?:\n\n|$)',
            # Chinese answer tags format
            r'答案[：:\s]*\n?(.*?)(?:\n\n|$)',
            # Bridge: or Bridges: format
            r'(?:Bridge|Bridges)[:\s]*\n?(.*?)(?:\n\n|$)',
            # Solution in backticks
            r'```(?:text|bridge|bridges)?\s*(.*?)\s*```',
            # Final answer format
            r'(?:Final answer|最终答案)[:\s]*\n?(.*?)(?:\n\n|$)',
        ]

        extracted_content = ""

        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, model_output, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_content = match.group(1).strip()
                # Handle escaped newlines in JSON format
                if '\\n' in extracted_content:
                    extracted_content = extracted_content.replace('\\n', '\n')
                # Remove extra quotes if present
                extracted_content = extracted_content.strip('"\'')
                break

        # If no tags found, search for bridge patterns directly in the entire output
        if not extracted_content:
            bridge_lines = []
            lines = model_output.split('\n')

            for line in lines:
                line = line.strip()
                # Match bridge patterns with various formats
                if self._is_bridge_line(line):
                    bridge_lines.append(line)

            extracted_content = '\n'.join(bridge_lines)

        # Clean and normalize the extracted content
        return self._normalize_bridge_format(extracted_content)

    def _is_bridge_line(self, line: str) -> bool:
        """Check if a line contains a valid bridge specification"""
        if not line or len(line.strip()) == 0:
            return False

        # Various bridge patterns - more comprehensive
        bridge_patterns = [
            # Standard format: (x1,y1)-(x2,y2):count
            r'\(\s*\d+\s*,\s*\d+\s*\)\s*[-–—]\s*\(\s*\d+\s*,\s*\d+\s*\)\s*[:\s]*\s*\d+',
            # With "to" connector: (x1,y1) to (x2,y2): count
            r'\(\s*\d+\s*,\s*\d+\s*\)\s*(?:to|TO)\s*\(\s*\d+\s*,\s*\d+\s*\)\s*[:\s]*\s*\d+',
            # With arrow: (x1,y1) → (x2,y2): count
            r'\(\s*\d+\s*,\s*\d+\s*\)\s*[→>]\s*\(\s*\d+\s*,\s*\d+\s*\)\s*[:\s]*\s*\d+',
            # With bridge prefix: bridge: (x1,y1)-(x2,y2):count
            r'(?:bridge|Bridge|BRIDGE)\s*[:\s]*\s*\(\s*\d+\s*,\s*\d+\s*\)\s*'
            r'[-–—]\s*\(\s*\d+\s*,\s*\d+\s*\)\s*[:\s]*\s*\d+',
            # Connect format: connect (x1,y1) to (x2,y2) with count bridge(s)
            r'(?:connect|Connect)\s*\(\s*\d+\s*,\s*\d+\s*\)\s*(?:to|TO)\s*'
            r'\(\s*\d+\s*,\s*\d+\s*\)\s*(?:with|using)\s*\d+',
            # Number at the end format: (x1,y1)-(x2,y2) count
            r'\(\s*\d+\s*,\s*\d+\s*\)\s*[-–—]\s*\(\s*\d+\s*,\s*\d+\s*\)\s+\d+',
        ]

        for pattern in bridge_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False

    def _normalize_bridge_format(self, content: str) -> str:
        """Normalize bridge specifications to standard format"""
        if not content:
            return ""

        lines = [line.strip() for line in content.split('\n') if line.strip()]
        normalized_lines = []

        for line in lines:
            # Extract coordinates and count using flexible regex patterns
            patterns = [
                # Standard format: (x1,y1)-(x2,y2):count
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[-–—]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                r'\s*[:\s]*\s*(\d+)',
                # With "to": (x1,y1) to (x2,y2): count
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:to|TO)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                r'\s*[:\s]*\s*(\d+)',
                # With arrow: (x1,y1) → (x2,y2): count
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[→>]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                r'\s*[:\s]*\s*(\d+)',
                # Connect format
                r'(?:connect|Connect)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:to|TO)\s*'
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:with|using)\s*(\d+)',
                # Number at end: (x1,y1)-(x2,y2) count
                r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[-–—]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+(\d+)',
            ]

            match = None
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    break

            if match:
                x1, y1, x2, y2, count = match.groups()
                # Validate count is a positive integer
                try:
                    count_int = int(count)
                    if count_int > 0:
                        normalized = f"({x1},{y1})-({x2},{y2}):{count_int}"
                        normalized_lines.append(normalized)
                except ValueError:
                    continue  # Skip invalid count

        return '\n'.join(normalized_lines)

    def evaluate(self, predicted_answer: str, ground_truth: str, initial_state: str) -> bool:
        """
        Evaluate if the predicted answer is correct based ONLY on game rules and initial state.
        The ground_truth parameter is ignored - evaluation is purely rule-based.

        Args:
            predicted_answer: Model's predicted bridge connections
            ground_truth: Reference answer (IGNORED - not used in evaluation)
            initial_state: JSON string or dict containing the puzzle's initial island configuration

        Returns:
            bool: True if the predicted answer satisfies all game rules
        """
        try:
            # Parse the initial state
            if isinstance(initial_state, str):
                puzzle_data = json.loads(initial_state)
            elif isinstance(initial_state, dict):
                puzzle_data = initial_state
            else:
                return False

            islands = puzzle_data.get('islands', [])
            if not islands:
                return False

            # Extract predicted answer if it hasn't been processed yet
            if not self._is_normalized_answer_format(predicted_answer):
                predicted_answer = self.extract_answer(predicted_answer)

            if not predicted_answer.strip():
                return False

            # Extract and parse predicted bridges
            predicted_bridges = self._parse_bridges(predicted_answer)

            if not predicted_bridges:
                return False

            # Validate the solution against all game rules
            return self._validate_solution(islands, predicted_bridges)

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return False

    def _is_normalized_answer_format(self, answer: str) -> bool:
        """Check if the answer is already in normalized format"""
        if not answer or not isinstance(answer, str):
            return False

        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        if not lines:
            return False

        # Check if all lines match the normalized format
        for line in lines:
            if not re.match(r'\(\d+,\d+\)-\(\d+,\d+\):\d+', line):
                return False
        return True

    def _parse_bridges(self, answer: str) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        """Parse bridge specifications into a list of tuples"""
        if not answer:
            return []

        bridges = []
        lines = [line.strip() for line in answer.split('\n') if line.strip()]

        for line in lines:
            match = re.match(r'\((\d+),(\d+)\)-\((\d+),(\d+)\):(\d+)', line)
            if match:
                try:
                    x1, y1, x2, y2, count = map(int, match.groups())
                    # Validate count is positive
                    if count > 0:
                        bridges.append(((x1, y1), (x2, y2), count))
                except ValueError:
                    continue  # Skip invalid lines

        return bridges

    def _validate_solution(self, islands: List[Dict], bridges: List[Tuple]) -> bool:
        """
        Validate the bridge solution against all game rules.

        Rules:
        1. Each island's bridge count must match its number
        2. Bridges must be horizontal or vertical
        3. No more than 2 bridges between any pair of islands
        4. Bridges cannot cross each other or pass through islands
        5. All islands must be connected in a single network
        6. Bridges can only connect existing islands
        """
        if not islands or not bridges:
            return False

        # Create island lookup by coordinates
        island_dict = {}
        island_positions = set()

        for island in islands:
            if not isinstance(island, dict):
                return False
            if 'x' not in island or 'y' not in island or 'num' not in island:
                return False
            try:
                x, y, num = int(island['x']), int(island['y']), int(island['num'])
                if num <= 0:  # Island number must be positive
                    return False
                island_dict[(x, y)] = num
                island_positions.add((x, y))
            except (ValueError, TypeError):
                return False

        # Rule validation
        island_bridge_counts = defaultdict(int)
        bridge_connections = defaultdict(int)  # Track bridges between pairs

        for (x1, y1), (x2, y2), count in bridges:
            # Rule 6: Check if both endpoints are valid islands
            if (x1, y1) not in island_positions or (x2, y2) not in island_positions:
                return False

            # Rule 2: Check if bridge is horizontal or vertical
            if not (x1 == x2 or y1 == y2):
                return False

            # Don't allow bridges to the same island
            if (x1, y1) == (x2, y2):
                return False

            # Count bridges for each island
            island_bridge_counts[(x1, y1)] += count
            island_bridge_counts[(x2, y2)] += count

            # Count bridges between pairs - normalize pair order
            pair = tuple(sorted([(x1, y1), (x2, y2)]))
            bridge_connections[pair] += count

        # Rule 1: Verify each island has correct number of bridges
        for pos, required_count in island_dict.items():
            if island_bridge_counts[pos] != required_count:
                return False

        # Rule 3: No more than 2 bridges between any pair
        for count in bridge_connections.values():
            if count > 2:
                return False

        # Rule 4: Check for bridge crossings and islands in path
        if not self._check_no_crossings(bridges, island_positions):
            return False

        # Rule 5: Check connectivity
        if not self._check_connectivity(island_positions, bridges):
            return False

        return True

    def _check_no_crossings(self, bridges: List[Tuple], island_positions: Set[Tuple[int, int]]) -> bool:
        """Check that bridges don't cross and don't pass through islands"""

        for i, bridge1 in enumerate(bridges):
            (x1a, y1a), (x1b, y1b), _ = bridge1

            # Check no islands in path (except endpoints)
            if x1a == x1b:  # Vertical bridge
                for y in range(min(y1a, y1b) + 1, max(y1a, y1b)):
                    if (x1a, y) in island_positions:
                        return False
            else:  # Horizontal bridge
                for x in range(min(x1a, x1b) + 1, max(x1a, x1b)):
                    if (x, y1a) in island_positions:
                        return False

            # Check for crossings with other bridges
            for j, bridge2 in enumerate(bridges[i + 1:], i + 1):
                if self._bridges_cross(bridge1, bridge2):
                    return False

        return True

    def _bridges_cross(self, bridge1: Tuple, bridge2: Tuple) -> bool:
        """Check if two bridges cross each other"""
        (x1a, y1a), (x1b, y1b), _ = bridge1
        (x2a, y2a), (x2b, y2b), _ = bridge2

        # One bridge is horizontal, other is vertical
        if x1a == x1b and y2a == y2b:  # bridge1 vertical, bridge2 horizontal
            # Check if they intersect
            if (min(y1a, y1b) < y2a < max(y1a, y1b)
                    and min(x2a, x2b) < x1a < max(x2a, x2b)):
                return True
        elif y1a == y1b and x2a == x2b:  # bridge1 horizontal, bridge2 vertical
            # Check if they intersect
            if (min(x1a, x1b) < x2a < max(x1a, x1b)
                    and min(y2a, y2b) < y1a < max(y2a, y2b)):
                return True

        return False

    def _check_connectivity(self, island_positions: Set[Tuple[int, int]], bridges: List[Tuple]) -> bool:
        """Check that all islands are connected through bridges"""
        if not island_positions:
            return True

        # Build adjacency graph
        graph = defaultdict(set)
        for (x1, y1), (x2, y2), _ in bridges:
            graph[(x1, y1)].add((x2, y2))
            graph[(x2, y2)].add((x1, y1))

        # DFS to check connectivity
        visited = set()
        start = next(iter(island_positions))
        stack = [start]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

        # All islands should be reachable
        return len(visited) == len(island_positions)
