import json
import re
from typing import Any, Dict, List, Set, Tuple, Union


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        raise NotImplementedError


class HitoriEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """
        准备提示，指导模型如何解答Hitori拼图

        Args:
            question: 问题描述
            params: 包含拼图数据的参数字典

        Returns:
            格式化的提示
        """
        grid = params.get("grid", [])
        grid_size = len(grid)

        prompt = f"{question}\n\n"
        prompt += "Hitori拼图说明：\n"
        prompt += "1. 需要将一些单元格涂黑\n"
        prompt += "2. 每行每列不能有重复的数字(在非黑色单元格中)\n"
        prompt += "3. 黑色单元格不能相邻(不能共享边)\n"
        prompt += "4. 所有白色单元格必须连通(通过上下左右移动)\n\n"

        prompt += f"网格尺寸: {grid_size}x{grid_size}\n"
        prompt += "拼图网格:\n"

        # 构建网格可视化
        for row in grid:
            prompt += " ".join(str(cell) for cell in row) + "\n"

        prompt += "\n请列出应该涂黑的单元格坐标，格式为(row, col)的集合。"
        prompt += "\n坐标从(0,0)开始计数，即左上角为(0,0)。"
        prompt += "\n请使用标准格式返回答案: {(r1, c1), (r2, c2), ...}"

        return prompt

    def extract_answer(self, model_output: str) -> Set[Tuple[int, int]]:
        """
        从模型输出中提取涂黑单元格的坐标集合，支持多种格式
        增强鲁棒性，处理各种边界情况

        Args:
            model_output: 模型生成的输出

        Returns:
            涂黑单元格的坐标集合
        """
        if not isinstance(model_output, str):
            return set()

        coords = set()

        # 预处理：移除常见的无关字符和噪音
        cleaned_output = re.sub(r'["""''`]', '', model_output)  # 移除引号
        cleaned_output = re.sub(r'\\n|\\t', ' ', cleaned_output)  # 移除转义字符

        # 方法1: 尝试匹配花括号包围的坐标集合
        # 支持格式: {(1,2), (3,4)} 或 {(1, 2), (3, 4)} 等
        brace_pattern = r'\{([^}]*)\}'
        brace_matches = re.findall(brace_pattern, cleaned_output)

        for match in brace_matches:
            # 在花括号内容中查找所有坐标对
            coord_matches = re.findall(r'[(\[]?\s*(\d+)\s*[,，]\s*(\d+)\s*[)\]]?', match)
            for row_str, col_str in coord_matches:
                try:
                    row, col = int(row_str), int(col_str)
                    if self._is_valid_coordinate(row, col):
                        coords.add((row, col))
                except ValueError:
                    continue

        # 方法2: 尝试匹配方括号包围的数组格式
        # 支持格式: [(1,2), (3,4)] 或 [[1,2], [3,4]]
        if not coords:
            bracket_patterns = [
                r'\[\s*([^]]*)\s*\]',  # 外层方括号
                r'\(\s*\[\s*([^]]*)\s*\]\s*\)',  # 圆括号包围的方括号
            ]

            for pattern in bracket_patterns:
                matches = re.findall(pattern, cleaned_output)
                for match in matches:
                    # 查找坐标对
                    coord_matches = re.findall(r'[(\[]?\s*(\d+)\s*[,，]\s*(\d+)\s*[)\]]?', match)
                    for row_str, col_str in coord_matches:
                        try:
                            row, col = int(row_str), int(col_str)
                            if self._is_valid_coordinate(row, col):
                                coords.add((row, col))
                        except ValueError:
                            continue

        # 方法3: 如果前面没找到，尝试直接在整个文本中查找坐标
        if not coords:
            # 增强的坐标识别模式
            coordinate_patterns = [
                r'\(\s*(\d+)\s*[,，]\s*(\d+)\s*\)',  # (1,2) 格式
                r'\[\s*(\d+)\s*[,，]\s*(\d+)\s*\]',  # [1,2] 格式
                r'(?:坐标|位置|cell|点)\s*[：:]\s*\(?(\d+)\s*[,，]\s*(\d+)\s*\)?',  # 坐标:1,2
                r'(?:row|行)\s*[=：:]\s*(\d+)\s*[,，\s]+(?:col|column|列)\s*[=：:]\s*(\d+)',  # row=1,col=2
                r'(?:col|column|列)\s*[=：:]\s*(\d+)\s*[,，\s]+(?:row|行)\s*[=：:]\s*(\d+)',  # col=2,row=1
                r'(?:第|第\s*)?(\d+)\s*行[，,\s]*(?:第|第\s*)?(\d+)\s*列',  # 第1行第2列
                r'(?:第|第\s*)?(\d+)\s*列[，,\s]*(?:第|第\s*)?(\d+)\s*行',  # 第2列第1行
            ]

            for pattern in coordinate_patterns:
                matches = re.findall(pattern, cleaned_output, re.IGNORECASE)
                for match in matches:
                    try:
                        if len(match) == 2:
                            # 处理行列可能互换的情况
                            if 'col' in pattern.lower() and pattern.index('col') < pattern.index('row'):
                                row, col = int(match[1]), int(match[0])
                            elif '列' in pattern and pattern.index('列') < pattern.index('行'):
                                row, col = int(match[1]), int(match[0])
                            else:
                                row, col = int(match[0]), int(match[1])

                            if self._is_valid_coordinate(row, col):
                                coords.add((row, col))
                    except (ValueError, AttributeError):
                        continue

        # 方法4: 尝试解析JSON格式
        if not coords:
            try:
                # 尝试找到JSON数组格式 [[1,2], [3,4]]
                json_patterns = [
                    r'\[\s*\[.*?\]\s*\]',  # [[1,2], [3,4]]
                    r'\[\s*\(.*?\)\s*\]',  # [(1,2), (3,4)]
                ]

                for json_pattern in json_patterns:
                    json_matches = re.findall(json_pattern, cleaned_output)
                    for json_str in json_matches:
                        try:
                            # 预处理JSON字符串
                            json_str = re.sub(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', r'[\1,\2]', json_str)
                            coord_list = json.loads(json_str)
                            if isinstance(coord_list, list):
                                for item in coord_list:
                                    if isinstance(item, list) and len(item) == 2:
                                        try:
                                            row, col = int(item[0]), int(item[1])
                                            if self._is_valid_coordinate(row, col):
                                                coords.add((row, col))
                                        except (ValueError, TypeError):
                                            continue
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue
            except Exception:
                pass

        # 方法5: 处理纯数字序列（如果有明确的分隔符）
        if not coords:
            # 匹配形如: 1,2 3,4 5,6 的格式
            number_pairs = re.findall(r'(\d+)\s*[,，]\s*(\d+)', cleaned_output)
            if number_pairs:
                # 只有在找到合理数量的坐标对时才使用这种方法
                valid_pairs = []
                for row_str, col_str in number_pairs:
                    try:
                        row, col = int(row_str), int(col_str)
                        if self._is_valid_coordinate(row, col):
                            valid_pairs.append((row, col))
                    except ValueError:
                        continue

                # 如果找到的坐标对数量合理（不会包含太多错误匹配）
                if 1 <= len(valid_pairs) <= 50:  # 假设合理的坐标数量范围
                    coords.update(valid_pairs)

        # 方法6: 处理表格形式的输出
        if not coords:
            # 查找表格形式的坐标列表
            table_patterns = [
                r'(?:shaded|black|涂黑|选择)[^:]*[:：]\s*([^.\n]+)',
                r'(?:answer|答案|结果)[^:]*[:：]\s*([^.\n]+)',
                r'(?:coordinates|坐标)[^:]*[:：]\s*([^.\n]+)',
            ]

            for pattern in table_patterns:
                matches = re.findall(pattern, cleaned_output, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    # 在匹配的内容中查找坐标
                    coord_matches = re.findall(r'[(\[]?\s*(\d+)\s*[,，]\s*(\d+)\s*[)\]]?', match)
                    for row_str, col_str in coord_matches:
                        try:
                            row, col = int(row_str), int(col_str)
                            if self._is_valid_coordinate(row, col):
                                coords.add((row, col))
                        except ValueError:
                            continue

        return coords

    def _is_valid_coordinate(self, row: int, col: int, max_size: int = 100) -> bool:
        """
        检查坐标是否有效

        Args:
            row, col: 坐标值
            max_size: 最大网格大小限制

        Returns:
            是否为有效坐标
        """
        return 0 <= row < max_size and 0 <= col < max_size

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        评估预测答案是否正确，仅根据initial_state和Hitori拼图规则验证
        不与ground_truth进行比较，只验证游戏规则

        Args:
            predicted_answer: 预测答案（可以是字符串、集合等格式）
            ground_truth: 真实答案（保留参数但不使用，仅为兼容性）
            initial_state: 初始网格状态（JSON字符串或二维数组）

        Returns:
            布尔值，表示预测答案是否满足游戏规则
        """
        # 提取预测答案中的坐标
        if isinstance(predicted_answer, str):
            pred_coords = self.extract_answer(predicted_answer)
        else:
            pred_coords = self._parse_coordinates(predicted_answer)

        if pred_coords is None:
            return False

        # 解析初始状态
        grid = self._parse_initial_state(initial_state)
        if grid is None:
            return False

        # 验证预测答案是否满足Hitori拼图规则
        return self._validate_hitori_rules(pred_coords, grid)

    def _parse_initial_state(self, initial_state: Any) -> List[List[int]]:
        """
        解析initial_state为网格，支持多种输入格式

        Args:
            initial_state: 初始状态（字符串、列表等）

        Returns:
            二维数组网格或None（如果解析失败）
        """
        try:
            # 如果已经是列表，直接使用
            if isinstance(initial_state, list):
                grid = initial_state
            else:
                # 尝试解析JSON格式的字符串
                grid = json.loads(str(initial_state))

            # 验证是否为有效的二维数组
            if (isinstance(grid, list) and len(grid) > 0 and all(isinstance(row, list) for row in grid)
                    and all(len(row) == len(grid[0]) for row in grid)
                    and all(isinstance(cell, (int, float)) for row in grid for cell in row)):

                # 确保所有元素都是整数
                return [[int(cell) for cell in row] for row in grid]

        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return None

    def _parse_coordinates(self, coords_input: Union[str, Set[Tuple[int, int]], List]) -> Set[Tuple[int, int]]:
        """
        将坐标输入解析为集合

        Args:
            coords_input: 坐标输入（字符串、集合或列表）

        Returns:
            坐标集合或None（如果解析失败）
        """
        # 如果已经是集合，验证格式
        if isinstance(coords_input, set):
            if all(isinstance(item, tuple) and len(item) == 2
                   and all(isinstance(coord, int) for coord in item) for item in coords_input):
                return coords_input
            return None

        # 如果是列表，尝试转换为集合
        if isinstance(coords_input, list):
            try:
                result = set()
                for item in coords_input:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        result.add((int(item[0]), int(item[1])))
                    else:
                        return None
                return result
            except (ValueError, TypeError):
                return None

        # 如果是字符串，使用extract_answer方法
        if isinstance(coords_input, str):
            return self.extract_answer(coords_input)

        return None

    def _validate_hitori_rules(self, shaded_cells: Set[Tuple[int, int]], grid: List[List[int]]) -> bool:
        """
        验证答案是否满足Hitori拼图的所有规则

        Args:
            shaded_cells: 涂黑单元格的坐标集合
            grid: 初始网格

        Returns:
            布尔值，表示答案是否有效
        """
        if not grid:
            return False

        size = len(grid)

        # 验证所有坐标都在网格范围内
        for r, c in shaded_cells:
            if not (0 <= r < size and 0 <= c < len(grid[r])):
                return False

        # 规则1：每行每列不能有重复的数字(在非黑色单元格中)
        if not self._check_no_duplicates(shaded_cells, grid):
            return False

        # 规则2：黑色单元格不能相邻
        if not self._check_no_adjacent_shaded(shaded_cells, size):
            return False

        # 规则3：所有白色单元格必须连通
        if not self._check_connectivity(shaded_cells, grid):
            return False

        return True

    def _check_no_duplicates(self, shaded_cells: Set[Tuple[int, int]], grid: List[List[int]]) -> bool:
        """
        检查每行每列在非黑色单元格中没有重复数字
        """
        size = len(grid)

        # 检查每行
        for i in range(size):
            row_values = []
            for j in range(size):
                if (i, j) not in shaded_cells:  # 如果不是黑色单元格
                    row_values.append(grid[i][j])

            if len(row_values) != len(set(row_values)):
                return False

        # 检查每列
        for j in range(size):
            col_values = []
            for i in range(size):
                if (i, j) not in shaded_cells:  # 如果不是黑色单元格
                    col_values.append(grid[i][j])

            if len(col_values) != len(set(col_values)):
                return False

        return True

    def _check_no_adjacent_shaded(self, shaded_cells: Set[Tuple[int, int]], size: int) -> bool:
        """
        检查黑色单元格不相邻
        """
        for r, c in shaded_cells:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            for nr, nc in neighbors:
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) in shaded_cells:
                    return False
        return True

    def _check_connectivity(self, shaded_cells: Set[Tuple[int, int]], grid: List[List[int]]) -> bool:
        """
        检查所有白色单元格连通
        """
        size = len(grid)

        # 获取所有白色单元格
        white_cells = []
        for r in range(size):
            for c in range(len(grid[r])):
                if (r, c) not in shaded_cells:
                    white_cells.append((r, c))

        if not white_cells:
            return False

        # 使用BFS检查连通性
        visited = set()
        queue = [white_cells[0]]
        visited.add(white_cells[0])

        while queue:
            r, c = queue.pop(0)
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            for nr, nc in neighbors:
                if (
                    0 <= nr < size and 0 <= nc < len(grid[nr])
                    and (nr, nc) not in shaded_cells
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return len(visited) == len(white_cells)
