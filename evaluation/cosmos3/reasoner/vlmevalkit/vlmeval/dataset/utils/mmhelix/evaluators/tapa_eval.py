import re
from collections import deque
from typing import Any, Dict, List, Tuple


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class TapaEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        initial_state = params.get('initial_state', '')
        size_info = params.get('size', '')

        prompt = f"""Please solve this Tapa puzzle following these rules:

1. All black cells must form a single connected group
2. No 2x2 block of black cells is allowed
3. Numbers in clue cells indicate lengths of connected black cell groups in the 8 surrounding cells
4. Provide your answer as coordinates of black cells in the format (row,column), separated by commas
5. Use (0,0) as the top-left corner

{question}

Initial state with clues:
{initial_state}

{f'Grid size: {size_info}' if size_info else ''}

Please provide your answer as coordinates of black cells in the format (row,column), separated by commas.
For example: (0,1), (1,2), (2,0), (2,1)"""

        return prompt

    def extract_answer(self, model_output: str) -> Any:
        if not model_output:
            return None

        text = model_output.strip()
        text = re.sub(r'```[a-z]*\n?', '', text)  # 移除markdown代码块标记
        text = re.sub(r'```', '', text)  # 移除结尾的```

        coordinates = self._extract_coordinates(text)
        if coordinates is not None:
            return {'type': 'coordinates', 'data': coordinates}

        grid = self._extract_grid(text)
        if grid is not None:
            return {'type': 'grid', 'data': grid}

        return None

    def _extract_coordinates(self, text: str) -> List[Tuple[int, int]]:
        coordinates = []
        coordinate_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches = re.findall(coordinate_pattern, text)

        for match in matches:
            try:
                x, y = int(match[0]), int(match[1])
                if 0 <= x <= 99 and 0 <= y <= 99:
                    coordinates.append((x, y))
            except (ValueError, IndexError):
                continue

        if coordinates:
            return coordinates

        alternative_patterns = [
            r'\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            r'\{\s*(\d+)\s*,\s*(\d+)\s*\}',
            r'(?:^|[^\d])\s*(\d+)\s*,\s*(\d+)\s*(?=[^\d]|$)',
        ]

        for pattern in alternative_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    try:
                        x, y = int(match[0]), int(match[1])
                        if 0 <= x <= 99 and 0 <= y <= 99:
                            coordinates.append((x, y))
                    except (ValueError, IndexError):
                        continue

                if coordinates:
                    return coordinates

        # 尝试寻找描述性格式
        descriptive_patterns = [
            r'(?:row|r)\s*(\d+)\s*(?:col|column|c)\s*(\d+)',
            r'(?:position|pos)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
        ]

        for pattern in descriptive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        x, y = int(match[0]), int(match[1])
                        if 0 <= x <= 99 and 0 <= y <= 99:
                            coordinates.append((x, y))
                    except (ValueError, IndexError):
                        continue

                if coordinates:
                    return coordinates

        return None if not coordinates else coordinates

    def _extract_grid(self, text: str) -> List[List[str]]:
        """提取网格格式的答案"""
        lines = text.split('\n')
        grid_lines = []

        # 各种可能的网格行模式
        patterns = [
            # 标准格式：直接的B/W/数字序列
            r'^[BWbw0-9\s]+$',
            # 包含逗号分隔的格式
            r'^[BWbw0-9\s,]+$',
            # 包含引号的格式
            r'^[BWbw0-9\s\'"]+$',
            # 包含下划线或点的格式
            r'^[BWbw0-9\s._-]+$'
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否匹配任何网格行模式
            for pattern in patterns:
                if re.match(pattern, line):
                    # 清理行内容：移除空格、逗号、引号等
                    clean_line = re.sub(r'[,\s\'"_-]', '', line)

                    # 标准化大小写
                    clean_line = clean_line.upper()

                    # 验证清理后的行只包含有效字符
                    if re.match(r'^[BW0-9]+$', clean_line) and len(clean_line) > 0:
                        grid_lines.append(clean_line)
                        break

        # 如果没有找到网格行，尝试更宽松的匹配
        if not grid_lines:
            # 寻找包含B/W字符的行
            for line in lines:
                line = line.strip()
                # 检查行中是否包含足够的B/W字符
                bw_count = len(re.findall(r'[BWbw]', line))
                if bw_count >= 3:  # 至少3个B/W字符才考虑
                    # 提取所有B/W/数字字符
                    extracted = re.findall(r'[BWbw0-9]', line)
                    if extracted:
                        clean_line = ''.join(extracted).upper()
                        grid_lines.append(clean_line)

        if not grid_lines:
            return None

        # 转换为二维网格
        grid = []
        for line in grid_lines:
            grid.append(list(line))

        # 验证网格是否为矩形
        if len(grid) == 0:
            return None

        expected_width = len(grid[0])
        for i, row in enumerate(grid):
            if len(row) != expected_width:
                # 尝试修复长度不一致的行
                if len(row) < expected_width:
                    # 如果行太短，用W补齐
                    row.extend(['W'] * (expected_width - len(row)))
                else:
                    # 如果行太长，截断
                    grid[i] = row[:expected_width]

        return grid

    def evaluate(self, predicted_answer: str, ground_truth: Any, initial_state: str) -> bool:
        """评估预测答案是否正确

        Args:
            predicted_answer: 模型的原始输出字符串
            ground_truth: 标准答案（本函数中将被忽略，仅根据规则验证）
            initial_state: 初始状态字符串，包含线索信息

        Returns:
            bool: 答案是否正确（仅基于游戏规则验证）
        """
        try:
            # 从模型输出中提取答案
            extracted = self.extract_answer(predicted_answer)

            if extracted is None:
                return False

            # 解析初始状态获取网格尺寸和线索
            initial_lines = initial_state.strip().split('\n')
            if not initial_lines:
                return False

            rows = len(initial_lines)
            cols = len(initial_lines[0]) if initial_lines else 0

            if rows == 0 or cols == 0:
                return False

            # 解析线索
            clues = self._parse_clues(initial_state)

            # 根据提取的答案类型进行处理
            if extracted['type'] == 'coordinates':
                # 坐标格式：将坐标转换为网格
                black_coordinates = extracted['data']
                grid = self._coordinates_to_grid(black_coordinates, rows, cols, clues)
            elif extracted['type'] == 'grid':
                # 网格格式：直接使用提取的网格
                grid = extracted['data']

                # 验证网格尺寸
                if len(grid) != rows or (grid and len(grid[0]) != cols):
                    return False

                # 检查线索位置是否保持一致
                if not self._check_clue_positions(grid, initial_state):
                    return False

                # 创建仅包含B/W的网格用于规则验证
                grid = self._create_bw_grid(grid, clues)
            else:
                return False

            if grid is None:
                return False

            # 验证所有Tapa规则
            result = self._verify_tapa_rules(grid, clues)
            return result

        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def _coordinates_to_grid(self, coordinates: List[Tuple[int, int]], rows: int, cols: int,
                             clues: Dict[Tuple[int, int], List[int]]) -> List[List[str]]:
        """将坐标列表转换为网格格式"""
        # 初始化网格，所有位置为白色
        grid = [['W' for _ in range(cols)] for _ in range(rows)]

        # 验证坐标是否在有效范围内
        for r, c in coordinates:
            if not (0 <= r < rows and 0 <= c < cols):
                return None

            # 检查坐标是否与线索位置冲突
            if (r, c) in clues:
                return None  # 坐标不能在线索位置

            grid[r][c] = 'B'

        return grid

    def _parse_clues(self, initial_state: str) -> Dict[Tuple[int, int], List[int]]:
        """解析初始状态中的线索"""
        clues = {}
        if not initial_state:
            return clues

        lines = initial_state.strip().split('\n')

        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char.isdigit():
                    # 单个数字线索
                    clues[(i, j)] = [int(char)]
                elif char in '0123456789':
                    # 确保是数字字符
                    clues[(i, j)] = [int(char)]

        return clues

    def _parse_clue_digits(self, digits_str: str) -> List[int]:
        """解析线索数字字符串 - 简化版本"""
        if len(digits_str) == 1:
            return [int(digits_str)]

        # 对于多位数字，每个数字都是单独的线索
        # 例如 "23" -> [2, 3], "123" -> [1, 2, 3]
        individual_digits = [int(d) for d in digits_str if d.isdigit()]
        return individual_digits

    def _check_clue_positions(self, grid: List[List[str]], initial_state: str) -> bool:
        """检查网格中线索位置是否与原始状态一致"""
        initial_lines = initial_state.strip().split('\n')

        if len(grid) != len(initial_lines):
            return False

        for i in range(len(grid)):
            if len(grid[i]) != len(initial_lines[i]):
                return False

            for j in range(len(grid[i])):
                initial_char = initial_lines[i][j]
                grid_char = grid[i][j]

                if initial_char.isdigit():
                    # 初始状态是数字，答案中也必须是相同的数字
                    if grid_char != initial_char:
                        return False
                elif initial_char == '.':
                    # 初始状态是空位，答案中必须是B或W
                    if grid_char not in ['B', 'W']:
                        return False

        return True

    def _create_bw_grid(self, grid: List[List[str]], clues: Dict[Tuple[int, int], List[int]]) -> List[List[str]]:
        """创建仅包含B/W的网格，将线索位置标记为W"""
        bw_grid = []

        # 获取所有线索位置
        clue_positions = set()
        for (row, col), numbers in clues.items():
            # 对于每个线索，标记其占用的位置
            clue_positions.add((row, col))

        for i in range(len(grid)):
            row = []
            for j in range(len(grid[i])):
                if (i, j) in clue_positions:
                    # 线索位置视为白色细胞
                    row.append('W')
                else:
                    row.append(grid[i][j])
            bw_grid.append(row)

        return bw_grid

    def _verify_tapa_rules(self, grid: List[List[str]], clues: Dict[Tuple[int, int], List[int]]) -> bool:
        """验证所有Tapa规则"""
        # 规则1: 检查所有黑色细胞是否形成单一连通组
        if not self._check_single_connected_group(grid):
            return False

        # 规则2: 检查是否存在2x2黑色块
        if self._has_2x2_black_block(grid):
            return False

        # 规则3: 验证所有线索约束
        if not self._verify_clues(grid, clues):
            return False

        # 规则4: 检查白色细胞的连通性（可选，取决于具体的Tapa变体）
        # 注释掉这个检查，因为在某些Tapa变体中白色细胞不需要连通
        # if not self._check_white_connectivity(grid):
        #     return False

        return True

    def _check_single_connected_group(self, grid: List[List[str]]) -> bool:
        """检查所有黑色细胞是否形成单一连通组"""
        rows, cols = len(grid), len(grid[0])
        black_cells = []

        # 找到所有黑色细胞
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 'B':
                    black_cells.append((i, j))

        if not black_cells:
            return True  # 没有黑色细胞也算有效

        # 从第一个黑色细胞开始BFS
        start = black_cells[0]
        visited = set([start])
        queue = deque([start])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < rows and 0 <= ny < cols
                        and (nx, ny) not in visited and grid[nx][ny] == 'B'):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) == len(black_cells)

    def _has_2x2_black_block(self, grid: List[List[str]]) -> bool:
        """检查是否存在2x2黑色块"""
        rows, cols = len(grid), len(grid[0])

        for i in range(rows - 1):
            for j in range(cols - 1):
                if (grid[i][j] == 'B' and grid[i][j + 1] == 'B'
                        and grid[i + 1][j] == 'B' and grid[i + 1][j + 1] == 'B'):
                    return True

        return False

    def _check_white_connectivity(self, grid: List[List[str]]) -> bool:
        """检查白色细胞是否连通"""
        rows, cols = len(grid), len(grid[0])
        white_cells = []

        # 找到所有白色细胞
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 'W':
                    white_cells.append((i, j))

        if not white_cells:
            return True  # 没有白色细胞也算有效

        # 从第一个白色细胞开始BFS
        start = white_cells[0]
        visited = set([start])
        queue = deque([start])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < rows and 0 <= ny < cols
                        and (nx, ny) not in visited and grid[nx][ny] == 'W'):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) == len(white_cells)

    def _verify_clues(self, grid: List[List[str]], clues: Dict[Tuple[int, int], List[int]]) -> bool:
        """验证所有线索约束"""
        for (clue_row, clue_col), expected_groups in clues.items():
            if not self._verify_single_clue(grid, clue_row, clue_col, expected_groups):
                return False
        return True

    def _verify_single_clue(self, grid: List[List[str]], clue_row: int, clue_col: int,
                            expected_groups: List[int]) -> bool:
        """验证单个线索约束"""
        rows, cols = len(grid), len(grid[0])

        # 获取线索位置周围8个邻居
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:  # 跳过线索细胞本身
                    continue
                ni, nj = clue_row + di, clue_col + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append((ni, nj))

        # 找到邻居中的黑色细胞
        black_neighbors = [(i, j) for i, j in neighbors if grid[i][j] == 'B']

        # 如果期望的组大小包含0，则应该没有黑色邻居
        if 0 in expected_groups:
            is_valid = len(black_neighbors) == 0

            return is_valid

        # 将黑色邻居分组为连通组
        groups = self._find_connected_groups_in_neighbors(black_neighbors)
        group_sizes = sorted([len(group) for group in groups])
        expected_sizes = sorted(expected_groups)

        is_valid = group_sizes == expected_sizes

        return is_valid

    def _find_connected_groups_in_neighbors(self, black_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """将邻居中的黑色细胞分组为连通组（包括对角连接）"""
        if not black_cells:
            return []

        cell_set = set(black_cells)
        visited = set()
        groups = []

        for cell in black_cells:
            if cell not in visited:
                group = []
                queue = deque([cell])
                visited.add(cell)

                while queue:
                    x, y = queue.popleft()
                    group.append((x, y))

                    # 检查8个方向的邻居（包括斜对角）
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (nx, ny) in cell_set and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny))

                groups.append(group)

        return groups
