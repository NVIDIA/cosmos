import ast
import json
import re
from typing import Any, Dict, List, Optional, Union


class CampsiteEvaluator:
    """
    Campsite puzzle evaluator that validates tent placement solutions.

    Rules:
    1. Each tent must be orthogonally adjacent to at least one tree
    2. No tents can be adjacent to each other, even diagonally
    3. The number of tents in each row and column must match the given constraints

    Coordinate System: 1-based indexing (top-left corner is [1,1])
    """

    def __init__(self):
        """Initialize the evaluator"""
        pass

    def extract_answer(self, model_output: str) -> Optional[List[List[int]]]:
        """
        从模型输出中提取答案坐标，具有极强的鲁棒性
        支持多种格式和边界情况
        """
        if not isinstance(model_output, str):
            # 如果输入已经是列表，直接验证格式
            if isinstance(model_output, list):
                return self._validate_coordinate_format(model_output)
            return None

        # 去除首尾空白字符和常见的干扰字符
        output = model_output.strip()
        output = output.replace('\n', ' ').replace('\t', ' ')
        output = re.sub(r'\s+', ' ', output)  # 合并多个空格

        # 方法1: 尝试直接解析完整的列表格式
        result = self._try_direct_parsing(output)
        if result is not None:
            return result

        # 方法2: 使用多种正则表达式模式匹配
        result = self._try_regex_patterns(output)
        if result is not None:
            return result

        # 方法3: 提取所有数字并尝试配对
        result = self._try_number_pairing(output)
        if result is not None:
            return result

        # 方法4: 处理特殊格式和语言描述
        result = self._try_special_formats(output)
        if result is not None:
            return result

        # 方法5: 尝试更激进的数字提取
        result = self._try_aggressive_extraction(output)
        if result is not None:
            return result

        # 方法6: 处理混合格式和特殊边界情况
        result = self._try_mixed_formats(output)
        if result is not None:
            return result

        # 如果所有方法都失败，返回None
        return None

    def _validate_coordinate_format(self, coords: List) -> Optional[List[List[int]]]:
        """验证坐标格式是否正确"""
        try:
            if not isinstance(coords, list):
                return None

            result = []
            for coord in coords:
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    try:
                        row, col = int(coord[0]), int(coord[1])
                        if row > 0 and col > 0:  # 1-based indexing
                            result.append([row, col])
                        else:
                            return None
                    except (ValueError, TypeError):
                        return None
                else:
                    return None

            return result if result else []  # 允许空列表
        except Exception:
            return None

    def _try_direct_parsing(self, output: str) -> Optional[List[List[int]]]:
        """尝试直接解析完整的列表格式"""
        try:
            # 清理常见的格式问题
            cleaned = output.strip()

            # 处理标准格式: [[1,2], [3,4]]
            if cleaned.startswith('[') and cleaned.endswith(']'):
                # 尝试直接解析
                try:
                    parsed = ast.literal_eval(cleaned)
                    return self._validate_coordinate_format(parsed)
                except Exception:
                    pass

                # 如果失败，尝试清理后再解析
                cleaned = re.sub(r'\s+', ' ', cleaned)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = re.sub(r'\[\s*,', '[', cleaned)
                cleaned = re.sub(r',,+', ',', cleaned)  # 移除多余的逗号
                try:
                    parsed = ast.literal_eval(cleaned)
                    return self._validate_coordinate_format(parsed)
                except Exception:
                    pass

            # 处理JSON格式
            try:
                parsed = json.loads(cleaned)
                return self._validate_coordinate_format(parsed)
            except Exception:
                pass

        except Exception:
            pass
        return None

    def _try_regex_patterns(self, output: str) -> Optional[List[List[int]]]:
        """使用多种正则表达式模式匹配坐标"""
        patterns = [
            # 标准格式: [[1,2], [3,4]]
            r'\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\](?:\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*\])*\s*\]',
            # 元组格式: [(1,2), (3,4)]
            r'\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\))*\s*\]',
            # 混合格式: [1,2], [3,4] (无外层括号)
            r'\[\s*(\d+)\s*,\s*(\d+)\s*\](?:\s*,?\s*\[\s*(\d+)\s*,\s*(\d+)\s*\])*',
            # 元组格式: (1,2), (3,4) (无外层括号)
            r'\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:\s*,?\s*\(\s*(\d+)\s*,\s*(\d+)\s*\))*',
        ]

        for pattern in patterns:
            try:
                # 使用findall找到所有数字对
                numbers = re.findall(r'(\d+)\s*,\s*(\d+)', output)
                if numbers:
                    coords = []
                    for num_pair in numbers:
                        try:
                            row, col = int(num_pair[0]), int(num_pair[1])
                            if row > 0 and col > 0:  # 验证1-based indexing
                                coords.append([row, col])
                        except (ValueError, TypeError):
                            continue

                    if len(coords) >= 0:  # 允许空答案
                        return coords
            except Exception:
                continue

        return None

    def _try_number_pairing(self, output: str) -> Optional[List[List[int]]]:
        """提取所有数字并尝试配对成坐标"""
        try:
            # 提取所有数字
            numbers = re.findall(r'\d+', output)
            if len(numbers) >= 2 and len(numbers) % 2 == 0:
                coords = []
                for i in range(0, len(numbers), 2):
                    try:
                        row, col = int(numbers[i]), int(numbers[i + 1])
                        if row > 0 and col > 0:  # 验证1-based indexing
                            coords.append([row, col])
                    except (ValueError, TypeError, IndexError):
                        continue

                # 如果提取的坐标数量合理（通常0-30个帐篷）
                if 0 <= len(coords) <= 30:
                    return coords
            elif len(numbers) == 0:
                # 没有数字，可能是空答案
                return []
        except Exception:
            pass

        return None

    def _try_special_formats(self, output: str) -> Optional[List[List[int]]]:
        """处理特殊格式和语言描述"""
        special_patterns = [
            # 处理 "坐标为: (1,2), (3,4)" 格式
            r'坐标[为是]?\s*[:：]?\s*(.+)',
            # 处理 "答案是: [[1,2], [3,4]]" 格式
            r'答案[是为]?\s*[:：]?\s*(.+)',
            # 处理 "positions are: (1,2), (3,4)" 格式
            r'positions?\s+(?:are|is)\s*[:：]?\s*(.+)',
            # 处理 "coordinates: [[1,2], [3,4]]" 格式
            r'coordinates?\s*[:：]?\s*(.+)',
            # 处理 "result: [[1,2], [3,4]]" 格式
            r'result\s*[:：]?\s*(.+)',
            # 处理 "answer: [[1,2], [3,4]]" 格式
            r'answer\s*[:：]?\s*(.+)',
            # 处理 "solution: [[1,2], [3,4]]" 格式
            r'solution\s*[:：]?\s*(.+)',
            # 处理 "tents: [[1,2], [3,4]]" 格式
            r'tents?\s*[:：]?\s*(.+)',
            # 处理 "final answer: [[1,2], [3,4]]" 格式
            r'final\s+answer\s*[:：]?\s*(.+)',
            # 处理 "tent coordinates: [[1,2], [3,4]]" 格式
            r'tent\s+coordinates?\s*[:：]?\s*(.+)',
            # 处理 "tent positions: [[1,2], [3,4]]" 格式
            r'tent\s+positions?\s*[:：]?\s*(.+)',
        ]

        for pattern in special_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                extracted_part = match.group(1).strip()
                # 递归调用其他方法处理提取的部分
                result = self._try_direct_parsing(extracted_part)
                if result is not None:
                    return result
                result = self._try_regex_patterns(extracted_part)
                if result is not None:
                    return result

        return None

    def _try_aggressive_extraction(self, output: str) -> Optional[List[List[int]]]:
        """更激进的数字提取方法，处理各种边界情况"""
        try:
            # 检查是否包含"空"、"无"、"none"等关键词
            empty_keywords = ['空', '无', 'none', 'empty', 'null', '没有', 'no tents', 'no tent', 'zero tent']
            for keyword in empty_keywords:
                if keyword.lower() in output.lower():
                    return []

            # 尝试找到任何看起来像坐标的模式
            # 模式：数字 数字 数字 数字... (连续的偶数个数字)
            number_sequences = re.findall(r'(?:\d+\s*,?\s*){2,}', output)
            for seq in number_sequences:
                numbers = re.findall(r'\d+', seq)
                if len(numbers) >= 2 and len(numbers) % 2 == 0:
                    coords = []
                    for i in range(0, len(numbers), 2):
                        try:
                            row, col = int(numbers[i]), int(numbers[i + 1])
                            if 1 <= row <= 30 and 1 <= col <= 30:  # 合理的范围
                                coords.append([row, col])
                        except Exception:
                            continue
                    if coords:
                        return coords

            # 查找行列描述格式："第1行第2列"
            chinese_pattern = r'第(\d+)行第(\d+)列'
            matches = re.findall(chinese_pattern, output)
            if matches:
                coords = []
                for match in matches:
                    try:
                        row, col = int(match[0]), int(match[1])
                        if row > 0 and col > 0:
                            coords.append([row, col])
                    except Exception:
                        continue
                if coords:
                    return coords

            # 查找行列描述格式："row 1 column 2"
            english_pattern = r'row\s+(\d+)\s+column\s+(\d+)'
            matches = re.findall(english_pattern, output, re.IGNORECASE)
            if matches:
                coords = []
                for match in matches:
                    try:
                        row, col = int(match[0]), int(match[1])
                        if row > 0 and col > 0:
                            coords.append([row, col])
                    except Exception:
                        continue
                if coords:
                    return coords

            # 查找 R1C1 格式（R代表行，C代表列）
            rc_pattern = r'R(\d+)C(\d+)'
            matches = re.findall(rc_pattern, output, re.IGNORECASE)
            if matches:
                coords = []
                for match in matches:
                    try:
                        row, col = int(match[0]), int(match[1])
                        if row > 0 and col > 0:
                            coords.append([row, col])
                    except Exception:
                        continue
                if coords:
                    return coords

        except Exception:
            pass

        return None

    def _try_mixed_formats(self, output: str) -> Optional[List[List[int]]]:
        """处理混合格式和特殊边界情况"""
        try:
            # 处理用分号、管道符等分隔的格式
            separators = [';', '|', '\n', '\t', '  ']
            for sep in separators:
                if sep in output:
                    parts = output.split(sep)
                    coords = []
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # 尝试从每个部分提取坐标
                        numbers = re.findall(r'\d+', part)
                        if len(numbers) == 2:
                            try:
                                row, col = int(numbers[0]), int(numbers[1])
                                if row > 0 and col > 0:
                                    coords.append([row, col])
                            except Exception:
                                continue
                    if coords:
                        return coords

            # 处理表格格式（尝试识别类似表格的结构）
            lines = output.split('\n')
            if len(lines) > 1:
                coords = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # 查找每行中的数字对
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) == 2:
                        try:
                            row, col = int(numbers[0]), int(numbers[1])
                            if row > 0 and col > 0:
                                coords.append([row, col])
                        except Exception:
                            continue
                if coords:
                    return coords

            # 处理单个数字对分隔的情况（没有括号）
            # 例如："1 2 3 4" -> [[1,2], [3,4]]
            numbers_only = re.findall(r'\b\d+\b', output)
            if len(numbers_only) >= 2 and len(numbers_only) % 2 == 0 and len(numbers_only) <= 60:  # 最多30个坐标
                coords = []
                for i in range(0, len(numbers_only), 2):
                    try:
                        row, col = int(numbers_only[i]), int(numbers_only[i + 1])
                        if 1 <= row <= 30 and 1 <= col <= 30:
                            coords.append([row, col])
                    except Exception:
                        continue
                # 只有当所有数字都能成功转换为合理坐标时才返回
                if len(coords) == len(numbers_only) // 2:
                    return coords

        except Exception:
            pass

        return None

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Union[str, Dict]) -> bool:
        """
        验证预测答案是否正确，仅基于initial_state和游戏规则进行判断

        Args:
            predicted_answer: 模型的预测答案（可以是字符串或列表）
            ground_truth: 标准答案（不使用，仅为接口兼容性保留）
            initial_state: 初始状态包含网格和约束条件

        Returns:
            bool: 答案是否正确

        Note:
            此函数完全不依赖ground_truth参数，仅根据游戏规则和约束条件验证答案正确性
        """
        try:
            # 解析初始状态
            if isinstance(initial_state, str):
                state = json.loads(initial_state)
            else:
                state = initial_state

            # 提取游戏状态
            input_grid = state["input_grid"]
            row_constraints = state["row_constraints"]
            col_constraints = state["col_constraints"]

            # 提取预测答案中的坐标
            if isinstance(predicted_answer, str):
                predicted_coords = self.extract_answer(predicted_answer)
                if predicted_coords is None:
                    return False
            else:
                predicted_coords = self._validate_coordinate_format(predicted_answer)
                if predicted_coords is None:
                    return False

            # 验证游戏规则
            return self._validate_game_rules(predicted_coords, input_grid, row_constraints, col_constraints)

        except Exception:
            return False

    def _validate_game_rules(
            self, tent_coords: List[List[int]], input_grid: List[List[str]],
            row_constraints: List[int], col_constraints: List[int]) -> bool:
        """
        验证帐篷位置是否符合游戏规则

        Args:
            tent_coords: 帐篷坐标列表 (1-based indexing)
            input_grid: 游戏网格
            row_constraints: 行约束
            col_constraints: 列约束

        Returns:
            bool: 是否符合所有规则
        """
        if not tent_coords:
            # 检查是否所有约束都为0
            return all(c == 0 for c in row_constraints) and all(c == 0 for c in col_constraints)

        rows, cols = len(input_grid), len(input_grid[0])

        # 转换为0-based索引用于内部处理
        tent_positions_0based = set()
        for coord in tent_coords:
            row, col = coord[0] - 1, coord[1] - 1  # 转换为0-based
            tent_positions_0based.add((row, col))

        # 规则验证1：检查所有帐篷位置是否在网格边界内
        for row, col in tent_positions_0based:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                return False

        # 规则验证2：每个帐篷必须与至少一棵树正交相邻
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        for row, col in tent_positions_0based:
            adjacent_to_tree = False
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if input_grid[nr][nc] == 'T':
                        adjacent_to_tree = True
                        break
            if not adjacent_to_tree:
                return False

        # 规则验证3：帐篷之间不能相邻（包括对角线）
        all_directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for row, col in tent_positions_0based:
            for dr, dc in all_directions:
                nr, nc = row + dr, col + dc
                if (nr, nc) in tent_positions_0based:
                    return False

        # 规则验证4：检查行约束
        actual_row_counts = [0] * rows
        for row, col in tent_positions_0based:
            actual_row_counts[row] += 1

        if actual_row_counts != row_constraints:
            return False

        # 规则验证5：检查列约束
        actual_col_counts = [0] * cols
        for row, col in tent_positions_0based:
            actual_col_counts[col] += 1

        if actual_col_counts != col_constraints:
            return False

        return True

    def get_detailed_feedback(
            self, predicted_answer: Any, ground_truth: Any,
            initial_state: Union[str, Dict]) -> Dict[str, Any]:
        """
        获取详细的验证反馈，用于调试和分析

        Returns:
            Dict包含详细的验证结果和错误信息
        """
        feedback = {
            "is_correct": False,
            "extracted_coords": None,
            "extraction_successful": False,
            "rule_violations": [],
            "error_message": None
        }

        try:
            # 解析初始状态
            if isinstance(initial_state, str):
                state = json.loads(initial_state)
            else:
                state = initial_state

            input_grid = state["input_grid"]
            row_constraints = state["row_constraints"]
            col_constraints = state["col_constraints"]
            rows, cols = len(input_grid), len(input_grid[0])

            # 提取答案
            if isinstance(predicted_answer, str):
                predicted_coords = self.extract_answer(predicted_answer)
            else:
                predicted_coords = self._validate_coordinate_format(predicted_answer)

            feedback["extracted_coords"] = predicted_coords
            feedback["extraction_successful"] = predicted_coords is not None

            if predicted_coords is None:
                feedback["error_message"] = "无法从答案中提取有效的坐标"
                return feedback

            # 详细规则验证
            if not predicted_coords:
                # 空答案的情况
                all_constraints_zero = (all(c == 0 for c in row_constraints)
                                        and all(c == 0 for c in col_constraints))
                if not all_constraints_zero:
                    feedback["rule_violations"].append("空答案但约束条件不为0")
                feedback["is_correct"] = all_constraints_zero
                return feedback

            tent_positions_0based = set()
            for coord in predicted_coords:
                row, col = coord[0] - 1, coord[1] - 1
                tent_positions_0based.add((row, col))

            # 检查边界
            for row, col in tent_positions_0based:
                if row < 0 or row >= rows or col < 0 or col >= cols:
                    feedback["rule_violations"].append(f"帐篷位置 ({row+1}, {col+1}) 超出网格边界")

            # 检查树邻接
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for row, col in tent_positions_0based:
                adjacent_to_tree = False
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols and input_grid[nr][nc] == 'T':
                        adjacent_to_tree = True
                        break
                if not adjacent_to_tree:
                    feedback["rule_violations"].append(f"帐篷 ({row+1}, {col+1}) 没有与树相邻")

            # 检查帐篷相邻
            all_directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for row, col in tent_positions_0based:
                for dr, dc in all_directions:
                    nr, nc = row + dr, col + dc
                    if (nr, nc) in tent_positions_0based:
                        feedback["rule_violations"].append(f"帐篷 ({row+1}, {col+1}) 与帐篷 ({nr+1}, {nc+1}) 相邻")

            # 检查行约束
            actual_row_counts = [0] * rows
            for row, col in tent_positions_0based:
                actual_row_counts[row] += 1

            for i, (actual, expected) in enumerate(zip(actual_row_counts, row_constraints)):
                if actual != expected:
                    feedback["rule_violations"].append(f"第{i+1}行帐篷数量错误: 实际{actual}, 期望{expected}")

            # 检查列约束
            actual_col_counts = [0] * cols
            for row, col in tent_positions_0based:
                actual_col_counts[col] += 1

            for i, (actual, expected) in enumerate(zip(actual_col_counts, col_constraints)):
                if actual != expected:
                    feedback["rule_violations"].append(f"第{i+1}列帐篷数量错误: 实际{actual}, 期望{expected}")

            feedback["is_correct"] = len(feedback["rule_violations"]) == 0

        except Exception as e:
            feedback["error_message"] = f"验证过程中发生错误: {str(e)}"

        return feedback
