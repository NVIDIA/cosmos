import re
from typing import Any, Dict


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: str,
                 params: Dict[str, Any] = None) -> bool:
        raise NotImplementedError


class NibblesEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        pass

    def extract_answer(self, model_output: str) -> str:
        """Extract movement directions from model output"""
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, model_output, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            content = model_output.strip()  # Fallback to full output if no tags found

        # Extract valid movement directions
        directions = []
        words = content.lower().split()
        valid_moves = {'up', 'down', 'left', 'right'}

        for word in words:
            if word in valid_moves:
                directions.append(word)

        return ' '.join(directions)

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate Snake game solution by simulating the game

        Args:
            predicted_answer: String of movement directions (e.g., "up down left right")
            ground_truth: Expected answer (for reference, but we verify by game simulation)
            initial_state: String containing grid size, snake position, direction, and apples
        """
        try:
            # Parse initial state
            game_state = self._parse_initial_state(initial_state)
            if not game_state:
                return False

            # Strictly validate that provided Direction matches snake geometry
            if not self._is_direction_consistent_with_snake(game_state):
                return False

            # Parse predicted moves
            if isinstance(predicted_answer, str):
                predicted_moves = predicted_answer.strip().split()
            else:
                predicted_moves = str(predicted_answer).strip().split()

            # Validate moves
            valid_moves = {'up', 'down', 'left', 'right'}
            predicted_moves = [move.lower() for move in predicted_moves if move.lower() in valid_moves]

            if not predicted_moves:
                return False

            # Simulate the game
            result = self._simulate_snake_game(game_state, predicted_moves)

            if result['success']:
                return True
            else:
                return False

        except Exception:
            return False

    def _parse_initial_state(self, initial_state: str) -> dict:
        """Parse the initial state string to extract game information"""
        try:
            lines = initial_state.strip().split('\n')
            game_info = {}

            for line in lines:
                line = line.strip()
                if line.startswith('Grid:'):
                    # Parse grid size, e.g., "Grid: 7x6"
                    size_str = line.split(':')[1].strip()
                    if 'x' in size_str:
                        rows, cols = map(int, size_str.split('x'))
                        game_info['rows'] = rows
                        game_info['cols'] = cols

                elif line.startswith('Snake:'):
                    # Parse snake positions, e.g., "Snake: (4,2) (4,1)"
                    coords_str = line.split(':')[1].strip()
                    coords_pattern = r'\((\d+),(\d+)\)'
                    matches = re.findall(coords_pattern, coords_str)
                    game_info['snake'] = [(int(r), int(c)) for r, c in matches]

                elif line.startswith('Direction:'):
                    # Parse initial direction, e.g., "Direction: left"
                    direction = line.split(':')[1].strip().lower()
                    game_info['direction'] = direction

                elif line.startswith('Apples:'):
                    # Parse apple positions, e.g., "Apples: (5,4)"
                    coords_str = line.split(':')[1].strip()
                    coords_pattern = r'\((\d+),(\d+)\)'
                    matches = re.findall(coords_pattern, coords_str)
                    game_info['apples'] = set((int(r), int(c)) for r, c in matches)

                elif line.startswith('Goal:'):
                    # Parse goal information for validation
                    goal_str = line.split(':')[1].strip()
                    # Extract number of apples to eat
                    num_match = re.search(r'(\d+)', goal_str)
                    if num_match:
                        game_info['target_apples'] = int(num_match.group(1))

            # Validate required fields
            required_fields = ['rows', 'cols', 'snake', 'direction', 'apples']
            if all(field in game_info for field in required_fields):
                return game_info
            else:
                return None

        except Exception:
            return None

    def _simulate_snake_game(self, game_state: dict, moves: list) -> dict:
        """
        Simulate the Snake game with given moves

        Returns:
            dict with success status, error message, and statistics
        """
        # Initialize game state
        rows, cols = game_state['rows'], game_state['cols']
        snake = list(game_state['snake'])  # List of (row, col) positions
        apples = set(game_state['apples'])  # Set of (row, col) positions
        total_apples = len(apples)
        apples_eaten = 0

        # Use provided initial direction strictly
        direction = game_state['direction']

        # Direction vectors
        direction_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        # Opposite directions (can't reverse)
        opposites = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }

        for i, move in enumerate(moves):
            # Check if trying to reverse direction
            if move == opposites.get(direction):
                return {
                    'success': False,
                    'error': f"Invalid move at step {i+1}: Cannot reverse direction from {direction} to {move}",
                    'apples_eaten': apples_eaten,
                    'total_apples': total_apples
                }

            # Update direction
            direction = move

            # Calculate new head position
            head_row, head_col = snake[0]
            dr, dc = direction_vectors[direction]
            new_head = (head_row + dr, head_col + dc)

            # Check bounds
            if not (0 <= new_head[0] < rows and 0 <= new_head[1] < cols):
                return {
                    'success': False,
                    'error': f"Hit boundary at step {i+1}: position {new_head} out of bounds ({rows}x{cols})",
                    'apples_eaten': apples_eaten,
                    'total_apples': total_apples
                }

            # Check self-collision
            if new_head in snake:
                return {
                    'success': False,
                    'error': f"Self-collision at step {i+1}: position {new_head} already occupied by snake",
                    'apples_eaten': apples_eaten,
                    'total_apples': total_apples
                }

            # Move snake
            snake.insert(0, new_head)

            # Check if apple eaten
            if new_head in apples:
                apples.remove(new_head)
                apples_eaten += 1
                # Snake grows (don't remove tail)
            else:
                # No apple eaten, remove tail
                snake.pop()

        # Check if all apples eaten
        if apples_eaten == total_apples:
            return {
                'success': True,
                'error': None,
                'apples_eaten': apples_eaten,
                'total_apples': total_apples
            }
        else:
            return {
                'success': False,
                'error': f"Not all apples eaten: {apples_eaten}/{total_apples}",
                'apples_eaten': apples_eaten,
                'total_apples': total_apples
            }

    def _is_direction_consistent_with_snake(self, game_state: dict) -> bool:
        """Check that the provided initial direction matches the orientation implied by the first two snake segments."""
        snake = game_state.get('snake', [])
        provided = game_state.get('direction')
        if not snake or len(snake) < 2:
            # With length 1, we cannot infer; accept provided
            return provided in {'up', 'down', 'left', 'right'}
        head_row, head_col = snake[0]
        neck_row, neck_col = snake[1]
        dr, dc = head_row - neck_row, head_col - neck_col
        if dr == -1 and dc == 0:
            inferred = 'up'
        elif dr == 1 and dc == 0:
            inferred = 'down'
        elif dr == 0 and dc == -1:
            inferred = 'left'
        elif dr == 0 and dc == 1:
            inferred = 'right'
        else:
            # Non-adjacent segments; invalid geometry
            return False
        return inferred == provided
