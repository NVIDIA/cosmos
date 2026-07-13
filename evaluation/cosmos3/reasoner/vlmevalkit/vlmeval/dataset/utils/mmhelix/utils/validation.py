import re


def puzzle_15_check(initial_state, model_input):
    if isinstance(initial_state, str):
        try:
            import ast
            initial_state = ast.literal_eval(initial_state)
        except Exception:
            return False

    if isinstance(model_input, str):
        # First try to extract tile numbers
        try:
            moves_str = model_input.strip('()[]{}')
            moves = [int(x.strip()) for x in moves_str.split(',') if x.strip()]
            moves_type = "tiles"
        except ValueError:
            # If not number tiles, try to extract directions using regex
            # This pattern matches "up", "down", "left", "right" regardless of surrounding characters
            direction_pattern = re.compile(r'\b(up|down|left|right)\b', re.IGNORECASE)
            directions = direction_pattern.findall(model_input.lower())

            if directions:
                moves = directions
                moves_type = "directions"
            else:
                return False
    else:
        moves = model_input
        moves_type = "tiles"  # Default to number tiles

    target_state = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]

    current_state = [row[:] for row in initial_state]

    # Find the empty position
    empty_pos = None
    for i in range(4):
        for j in range(4):
            if current_state[i][j] == 0:
                empty_pos = (i, j)
                break
        if empty_pos:
            break

    move_history = []

    for move in moves:
        if moves_type == "tiles":
            # Original number tile movement logic
            num_pos = None
            for i in range(4):
                for j in range(4):
                    if current_state[i][j] == move:
                        num_pos = (i, j)
                        break
                if num_pos:
                    break

            if not num_pos:
                return False

            row_diff = abs(num_pos[0] - empty_pos[0])
            col_diff = abs(num_pos[1] - empty_pos[1])

            if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
                return False

            current_state[empty_pos[0]][empty_pos[1]] = move
            current_state[num_pos[0]][num_pos[1]] = 0
            empty_pos = num_pos

        elif moves_type == "directions":
            # Direction movement logic
            new_pos = None

            if move == "up":
                # Move the tile below the empty space upward
                if empty_pos[0] < 3:  # Ensure there's a row below
                    new_pos = (empty_pos[0] + 1, empty_pos[1])
            elif move == "down":
                # Move the tile above the empty space downward
                if empty_pos[0] > 0:  # Ensure there's a row above
                    new_pos = (empty_pos[0] - 1, empty_pos[1])
            elif move == "left":
                # Move the tile to the right of the empty space leftward
                if empty_pos[1] < 3:  # Ensure there's a column to the right
                    new_pos = (empty_pos[0], empty_pos[1] + 1)
            elif move == "right":
                # Move the tile to the left of the empty space rightward
                if empty_pos[1] > 0:  # Ensure there's a column to the left
                    new_pos = (empty_pos[0], empty_pos[1] - 1)

            if not new_pos:
                return False

            # Swap empty space and the tile
            tile_value = current_state[new_pos[0]][new_pos[1]]
            current_state[empty_pos[0]][empty_pos[1]] = tile_value
            current_state[new_pos[0]][new_pos[1]] = 0
            empty_pos = new_pos

        move_history.append((move, [row[:] for row in current_state]))

    return current_state == target_state


def format_state(state):
    result = ""
    for row in state:
        result += " ".join(f"{num:2d}" for num in row) + "\n"
    return result


def print_state(state):
    for row in state:
        print(" ".join(f"{num:2d}" for num in row))
    print()


def hanoi_check(initial_state, answer):
    pegs = [list(peg) for peg in initial_state]

    # Use regex to extract all coordinate pairs in different formats
    # This pattern matches (disk,dest) or (disk dest) with optional spaces
    move_pattern = re.compile(r'\(\s*(\d+)\s*[,\s]\s*(\d+)\s*\)')
    matches = move_pattern.findall(answer)

    if not matches:
        return False

    moves = [(int(disk), int(dest)) for disk, dest in matches]

    for disk, dest_peg in moves:
        if dest_peg <= 0 or dest_peg > len(pegs):
            return False

        src_peg_idx = None
        for i, peg in enumerate(pegs):
            if disk in peg:
                src_peg_idx = i
                break

        if src_peg_idx is None:
            return False

        if pegs[src_peg_idx][-1] != disk:
            return False

        if src_peg_idx == dest_peg - 1:
            return False

        dest_peg_idx = dest_peg - 1
        if pegs[dest_peg_idx] and pegs[dest_peg_idx][-1] < disk:
            return False

        pegs[dest_peg_idx].append(pegs[src_peg_idx].pop())

    for i in range(len(pegs) - 1):
        if pegs[i]:
            return False

    return True


def maze_check(text_representation, response):

    maze = text_representation.strip().split('\n')

    start_position = None
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 'S':
                start_position = (i, j)
                break
        if start_position:
            break

    if not start_position:
        return False

    # Use regex to extract directions, case-insensitive and handle various separators
    direction_pattern = re.compile(r'\b(up|down|left|right)\b', re.IGNORECASE)
    directions = direction_pattern.findall(response.lower())

    if not directions:
        return False

    current_position = start_position

    for direction in directions:
        i, j = current_position
        if direction == "up":
            if i > 0 and maze[i - 1][j] == ' - ':
                return False
            i -= 2
        elif direction == "down":
            if i < len(maze) - 1 and maze[i + 1][j] == ' - ':
                return False
            i += 2
        elif direction == "left":
            if j > 0 and maze[i][j - 1] == ' | ':
                return False
            j -= 2
        elif direction == "right":
            if j < len(maze[i]) - 1 and maze[i][j + 1] == ' | ':
                return False
            j += 2
        else:
            return False

        if i < 0 or i >= len(maze) or j < 0 or j >= len(maze[i]):
            return False

        current_position = (i, j)

    i, j = current_position
    return maze[i][j] == 'E'
