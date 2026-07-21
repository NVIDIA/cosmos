import json
import re
from typing import Any, Dict, List, Optional


def _default_prompt_tpl(q: str) -> str:
    return (f"Please solve the Eulero puzzle. Return the grid in rows separated by newlines and cells by '|'.\n"
            f"Question:\n{q}\n<answer>...</answer>")


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(
        self,
        predicted_answer: Any,
        ground_truth: Any,
        initial_state: Any,
        params: Dict[str, Any] = None
    ) -> bool:
        raise NotImplementedError


class EuleroEvaluator(BaseEvaluator):
    # -----------------------
    # Public API
    # -----------------------
    def prepare_prompt(self, question: str, params: Dict[str, Any] = None) -> str:
        if params and isinstance(params, dict) and "prompt_template" in params:
            return str(params["prompt_template"]).format(question=question)
        try:
            from utils.constants import PROMPT_EULERO
            return PROMPT_EULERO.format(question)
        except Exception:
            return _default_prompt_tpl(question)

    def extract_answer(self, model_output: str) -> str:
        """
        从模型输出中提取网格（支持 <answer> 标签、任意空白、大小写、任意 N×N）。
        返回标准化的以 '|' 分隔、行以 '\n' 分隔的文本。
        """
        if not isinstance(model_output, str):
            model_output = str(model_output)

        # 优先从 <answer>...</answer> 中取
        m = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL | re.IGNORECASE)
        content = (m.group(1) if m else model_output).strip()

        # 若文本中存在明确的按行-按'|'分隔的结构，则直接规范化
        if self._looks_like_piped_grid(content):
            return self._normalize_grid(content)

        # 否则从全文提取 A1/B2 等对并按行聚合（尽力）
        pairs = re.findall(r"[A-Za-z]\d+", content)
        if not pairs:
            return ""

        # 估计网格大小（letters 与 numbers 的基数的 max）
        letters = {p[0].upper() for p in pairs}
        numbers = {p[1:] for p in pairs}
        n = max(len(letters), len(numbers))
        if n == 0:
            return ""

        expected = n * n
        if len(pairs) < expected:
            # 数据不足，尽力而为但返回空表示无法构出完整网格
            return ""

        # 取前 n^2 个，按行输出（大写化字母）
        canon_rows = []
        idx = 0
        for _ in range(n):
            row_pairs = [pairs[idx + j] for j in range(n)]
            row_pairs = [rp[0].upper() + rp[1:] for rp in row_pairs]
            canon_rows.append("|".join(row_pairs))
            idx += n
        return "\n".join(canon_rows)

    def evaluate(
        self,
        predicted_answer: str,
        ground_truth: Any,
        initial_state: Any,
        params: Dict[str, Any] = None
    ) -> bool:
        """
        规则：
        1) 每格均为合法的 Letter+Number；
        2) 每行/列 字母各出现一次；
        3) 每行/列 数字各出现一次；
        4) 全局 pair 唯一；
        5) 尊重预填（initial_state）。
        """
        # 1) 解析预测网格
        predicted_grid = self._parse_grid(predicted_answer)
        if not predicted_grid:
            return False

        n = len(predicted_grid)

        # 2) 解析/矫正 初始网格
        initial_grid = self._parse_initial_state(initial_state)

        if not initial_grid:
            # 解析不到时，默认同尺寸全 None
            initial_grid = [[None] * n for _ in range(n)]
        else:
            # 若尺寸不一致，尝试矫正到 n×n
            initial_grid = self._coerce_to_size(initial_grid, n)

        if len(initial_grid) != n or any(len(r) != n for r in initial_grid):
            return False

        # 3) 规则校验
        if not self._check_valid_pairs(predicted_grid, n):
            return False

        if not self._check_letters_unique_in_rows_and_columns(predicted_grid):
            return False

        if not self._check_numbers_unique_in_rows_and_columns(predicted_grid):
            return False

        if not self._check_unique_pairs(predicted_grid):
            return False

        if not self._check_respects_initial_state(predicted_grid, initial_grid):
            return False

        return True

    # -----------------------
    # Parsing helpers
    # -----------------------
    def _looks_like_piped_grid(self, s: str) -> bool:
        lines = [ln for ln in re.split(r"\r?\n", s) if ln.strip() != ""]
        return any("|" in ln for ln in lines)

    def _normalize_grid(self, grid_str: str) -> str:
        """
        规范化：去掉行首尾空白、删除纯空行、保留 '|' 分隔。
        """
        if not grid_str:
            return ""
        lines = [line.strip() for line in re.split(r"\r?\n", grid_str) if line.strip() != ""]
        return "\n".join(lines)

    def _parse_grid(self, grid_str: str) -> List[List[str]]:
        """
        解析预测解网格（严格：每行必须能解析出 A1、B2... 且每行等长）。
        """
        if not grid_str:
            return []

        normalized = self._normalize_grid(grid_str)
        if not normalized:
            return []

        rows: List[List[str]] = []
        for raw in normalized.split("\n"):
            # 优先用 '|' 分割，兼容无 '|' 时用正则抽取
            if "|" in raw:
                cells = [c.strip() for c in re.split(r"\s*\|\s*", raw)]
            else:
                cells = re.findall(r"[A-Za-z]\d+", raw)

            # 每个单元需是合法 pair
            parsed = []
            for cell in cells:
                if re.fullmatch(r"[A-Za-z]\d+", cell):
                    parsed.append(cell[0].upper() + cell[1:])  # 统一大写字母
                else:
                    # 预测解中出现非 pair，视为解析失败
                    parsed = []
                    break
            if not parsed:
                return []
            rows.append(parsed)

        # 行等长校验
        if not rows or not all(len(r) == len(rows[0]) for r in rows):
            return []
        return rows

    def _parse_initial_state(self, initial_state: Any) -> List[List[Optional[str]]]:
        """
        解析初始网格：优先逐格（'|'）解析，保留空格行与空单元。
        接受：
          - 带 '|' 的字符串（空位可为 '', '_', '__', '-', '--', ' '）
          - JSON 字符串（会递归解析）
          - 2D 列表（元素为 None/''/合法 pair）
        """
        # 2D 列表
        if isinstance(initial_state, list) and all(isinstance(r, list) for r in initial_state):
            return self._normalize_initial_list(initial_state)

        # 字符串
        if isinstance(initial_state, str):
            s = initial_state.strip("\n\r")

            # 1) 优先：逐格按 '|' 解析（保留空）
            lines = [ln.strip() for ln in s.splitlines() if True]  # 不丢空行
            if any("|" in ln for ln in lines):
                grid: List[List[Optional[str]]] = []
                for ln in lines:
                    cells = re.split(r"\s*\|\s*", ln.strip())
                    row: List[Optional[str]] = []
                    for cell in cells:
                        token = cell.strip()
                        if re.fullmatch(r"[A-Za-z]\d+", token):
                            row.append(token[0].upper() + token[1:])
                        elif token in {"", "_", "__", "-", "--", " "}:
                            row.append(None)
                        else:
                            # 未知记号按空处理
                            row.append(None)
                    if row:
                        grid.append(row)
                # 行长不一致先返回，后续会做尺寸矫正
                return grid if grid else []

            # 2) 尝试 JSON
            try:
                data = json.loads(s)
                return self._parse_initial_state(data)
            except Exception:
                pass

            # 3) 退回：像解析预测解那样的网格（但空位不可识别）
            # 若按此法，空行会被丢弃；建议仅作为最后兜底
            grid_like = self._parse_grid(s)
            if grid_like:
                # 将所有单元保留（预测式解析不会产生空），符合“全预填”的含义
                return grid_like

        # 其它类型或失败
        return []

    def _normalize_initial_list(self, grid: List[List[Any]]) -> List[List[Optional[str]]]:
        """
        将任意 2D 列表中的元素规范化为合法 pair 或 None。
        """
        norm: List[List[Optional[str]]] = []
        for row in grid:
            new_row: List[Optional[str]] = []
            for cell in row:
                if cell is None:
                    new_row.append(None)
                else:
                    token = str(cell).strip()
                    if re.fullmatch(r"[A-Za-z]\d+", token):
                        new_row.append(token[0].upper() + token[1:])
                    elif token in {"", "_", "__", "-", "--"}:
                        new_row.append(None)
                    else:
                        new_row.append(None)
            norm.append(new_row)
        return norm

    def _coerce_to_size(self, grid: List[List[Optional[str]]], n: int) -> List[List[Optional[str]]]:
        """
        将初始网格强制矫正为 n×n：
        - 行数不足：在末尾补 None 行；
        - 列数不足：各行右侧补 None；
        - 行/列超出：裁剪（并打印提示）。
        """
        rows = len(grid)
        if rows < n:
            grid = grid + [[None] * (max(len(r) for r in grid) if grid else n) for _ in range(n - rows)]
        elif rows > n:
            grid = grid[:n]

        # 统一列数到 n
        coerced: List[List[Optional[str]]] = []
        for r in grid:
            cols = len(r)
            if cols < n:
                coerced.append(r + [None] * (n - cols))
            elif cols > n:
                coerced.append(r[:n])
            else:
                coerced.append(r)
        return coerced

    # -----------------------
    # Rule checks
    # -----------------------
    def _check_valid_pairs(self, grid: List[List[str]], n: int) -> bool:
        valid_letters = {chr(ord('A') + i) for i in range(n)}
        valid_numbers = {str(i + 1) for i in range(n)}
        for row in grid:
            for cell in row:
                if not re.fullmatch(r"[A-Z]\d+", cell):
                    return False
                if cell[0] not in valid_letters or cell[1:] not in valid_numbers:
                    return False
        return True

    def _check_letters_unique_in_rows_and_columns(self, grid: List[List[str]]) -> bool:
        n = len(grid)
        # rows
        for row in grid:
            letters = [c[0] for c in row]
            if len(letters) != len(set(letters)):
                return False
        # cols
        for c in range(n):
            letters = [grid[r][c][0] for r in range(n)]
            if len(letters) != len(set(letters)):
                return False
        return True

    def _check_numbers_unique_in_rows_and_columns(self, grid: List[List[str]]) -> bool:
        n = len(grid)
        # rows
        for row in grid:
            nums = [c[1:] for c in row]
            if len(nums) != len(set(nums)):
                return False
        # cols
        for c in range(n):
            nums = [grid[r][c][1:] for r in range(n)]
            if len(nums) != len(set(nums)):
                return False
        return True

    def _check_unique_pairs(self, grid: List[List[str]]) -> bool:
        flat = [c for row in grid for c in row]
        return len(flat) == len(set(flat))

    def _check_respects_initial_state(
        self,
        predicted_grid: List[List[str]],
        initial_grid: List[List[Optional[str]]]
    ) -> bool:
        n = len(predicted_grid)
        for r in range(n):
            for c in range(n):
                v0 = initial_grid[r][c]
                if v0 is not None and v0 != "":
                    if predicted_grid[r][c] != v0:
                        return False
        return True
