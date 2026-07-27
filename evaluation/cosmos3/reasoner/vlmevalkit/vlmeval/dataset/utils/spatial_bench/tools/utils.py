import ast
import json
import math
import re
import string

import numpy as np
import pandas as pd

from vlmeval.smp.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# From mcq items to build choices texts
# ---------------------------------------------------------------------
def _clean_text(x) -> str:
    """Cast to str, collapse whitespace, strip."""
    s = str(x)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _nonempty(v) -> bool:
    """Return True if v is not None/NaN/empty (after strip for scalars)."""
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) > 0
    return str(v).strip() != ""


# Only whitespace + punctuation (no letters / digits / CJK)
_ONLY_PUNCT_WS_RE = re.compile(r'^[\s\.\,\;\:\!\?\-\_/\\\|\(\)\[\]\{\}【】（）·・、—]+$')


def _normalize_choice_body(raw) -> str:
    """
    Normalize choice text:
      - None / NaN -> ''
      - Collapse whitespace
      - If only whitespace/punctuation -> ''
      - Keep '<image>' etc. as-is
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ''
    txt = _clean_text(raw)
    if not txt or _ONLY_PUNCT_WS_RE.fullmatch(txt):
        return ''
    return txt


def _parse_candidates(val, max_letter: str = 'F'):
    """
    Parse 'options' / 'candidates' into a list of cleaned strings.
    """
    # 1) already a list
    if isinstance(val, list):
        return [_clean_text(x) for x in val]

    # 2) string cases
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None

        # 2a) try stringified Python list
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [_clean_text(x) for x in parsed]
        except Exception:
            # If parsing fails, fall back to extracting choices from the string.
            pass

        # 2b) fallback: treat as a mini "options block"
        mapping = extract_choices_from_question(s, max_letter=max_letter)
        if mapping:
            out = []
            for ch in string.ascii_uppercase:
                if ch in mapping:
                    out.append(_clean_text(mapping[ch]))
                else:
                    if ch == 'A':
                        continue
                    break
            return out or None

        return None

    # 3) other types -> not supported
    return None


def _letters_upto(max_letter: str = 'F', n: int | None = None):
    """
    Letter set helper.

    If n is given and > 0: ['A', ..., up to n].
    Else: ['A', ..., max_letter].
    """
    max_letter = max_letter.upper()
    all_letters = list(string.ascii_uppercase)
    if n is not None and isinstance(n, int) and n > 0:
        return all_letters[:min(n, 26)]
    end_idx = all_letters.index(max_letter) + 1
    return all_letters[:end_idx]


# Extract choices from explicit columns (A/B/C/...)
def _extract_from_columns(item, letter_set=('A', 'B', 'C', 'D', 'E', 'F')):
    """
    Read choices from columns A–F (or extended):

      - column exists & nonempty -> normalized text
      - column exists & empty    -> ''
      - column missing           -> no key
    """
    return {
        ch: (_normalize_choice_body(item[ch]) if _nonempty(item[ch]) else '')
        for ch in letter_set
        if ch in item
    }


# Extract choices from question text
def _slice_by_markers(text: str, markers):
    """
    Given label markers (letter, end_idx, start_idx), slice text into segments:

      [end_i : start_{i+1})  as body for letter_i,
      last marker goes to end of text.
    """
    out = {}
    markers = sorted(markers, key=lambda x: x[2])
    for i, (ch, end_label, start_pos) in enumerate(markers):
        next_start = markers[i + 1][2] if i + 1 < len(markers) else len(text)
        raw = text[end_label:next_start]
        out[ch] = _normalize_choice_body(raw)
    return out


def _contiguous_prefix_len(keys_iterable):
    """
    Count how many letters we have consecutively from 'A'.

    {'A','B','C'} -> 3
    {'B','C'}     -> 0
    {'A','C'}     -> 1
    """
    s = {k.upper() for k in keys_iterable}
    k = 0
    for ch in string.ascii_uppercase:
        if ch in s:
            k += 1
        else:
            break
    return k


def extract_choices_from_question(q: str, max_letter: str = 'F') -> dict:
    """
    Heuristically parse choices from question text.

    Tries:
      - Line-based labels (each option starts a line)
      - Inline labels ("A. xxx B. yyy C. zzz")

    Returns: {'A': '...', 'B': '...', ...} or {}.
    """
    if not isinstance(q, str) or not q.strip():
        return {}
    text = q

    # Drop preamble before "Options:" / "选项:"
    m = re.search(r'(?i)(options?|选项)\s*[:：]?', text)
    if m:
        text = text[m.end():]

    letters = ''.join(_letters_upto(max_letter))

    # ---------------- Line-based: e.g. each option on its own line ----------------
    LINE_LABEL = re.compile(
        rf'(?mi)^[ \t]*'
        rf'(?:[*_`>•·\-]+\s*)?'
        rf'(?:[\(\[\{{（【]\s*)?'
        rf'([{letters}])'                  # A / B / ...
        rf'(?:\s*[\)\]\}}）】])?'
        rf'\s*[\.．:：\)\]】、-]\s*'        # A. / A) / A: ...
    )

    from_lines: dict[str, str] = {}

    for m in LINE_LABEL.finditer(text):
        ch = m.group(1).upper()

        line_end = text.find('\n', m.end())
        if line_end == -1:
            line_end = len(text)
        raw = text[m.end():line_end]
        body = _normalize_choice_body(raw)
        from_lines[ch] = body

    # ---------------- Inline: e.g. "A. foo  B. bar  C. baz" ----------------
    INLINE_LABEL = re.compile(
        rf'(?<![A-Za-z0-9])'
        rf'([{letters}])'
        rf'\s*[\.．:：\)\]】、-]\s*'
        rf'(?![A-Za-z0-9])'
    )
    inline_markers = [(m.group(1).upper(), m.end(), m.start())
                      for m in INLINE_LABEL.finditer(text)]
    from_inline = _slice_by_markers(text, inline_markers) if inline_markers else {}

    # Pick better candidate
    ls_k = _contiguous_prefix_len(from_lines.keys())
    il_k = _contiguous_prefix_len(from_inline.keys())

    if ls_k > il_k:
        chosen = from_lines
    elif il_k > ls_k:
        chosen = from_inline
    else:
        if len(from_lines) > len(from_inline):
            chosen = from_lines
        elif len(from_inline) > len(from_lines):
            chosen = from_inline
        else:
            chosen = from_lines or from_inline

    return chosen


# Top-level: build_choices
def build_choices(item: dict, max_letter: str = 'F') -> dict:
    """
    Build a choice dict for one item (row-like mapping).

    Priority:
      1) 'options' / 'candidates' (options > candidates)
      2) Columns A..max_letter
      3) Parse from question text
      4) Fallback: {}
    """
    # 1) options / candidates
    seq = None
    for key in ('options', 'candidates'):
        if key in item:
            parsed = _parse_candidates(item[key])
            if parsed:
                seq = parsed
                break

    if seq:
        letters = _letters_upto(n=len(seq))
        return {ch: (seq[i] if i < len(seq) else '') for i, ch in enumerate(letters)}

    # 2) A–F (or extended) columns
    letters = _letters_upto(max_letter=max_letter)
    from_cols = _extract_from_columns(item, letter_set=letters)
    if from_cols:
        return from_cols

    # 3) From question text
    q = item.get('question')
    if isinstance(q, str):
        from_q = extract_choices_from_question(q, max_letter=max_letter)
        if from_q:
            return from_q

    # 4) Nothing found
    return {}


# ---------------------------------------------------------------------
# From spatial items to parse 2d points
# ---------------------------------------------------------------------
class Point2DParser:
    """
    Generic 2D point parser.

    - Parse model outputs into a set of (x, y) coordinates.
    - Support JSON / Python literals and text patterns.
    - First use _json2pts, then fall back to _text2pts.
    """

    _has_logged_hint = False

    @classmethod
    def log_hint(cls, task_name: str | None = None):
        if cls._has_logged_hint:
            return

        prefix = f'[{task_name}]' if task_name else '[Point2DParser]'
        msg = (
            f'{prefix} Using default Point2DParser:\n'
            '  - expects JSON / Python literal with "point_2d",\n'
            '    where coordinates may be:\n'
            '      * pixels (0 ~ W/H),\n'
            '      * [0, 1] normalized,\n'
            '      * [0, 1000] normalized (e.g., Qwen3-VL style);\n'
            '  - falls back to "(x, y)" or "(x0, y0, x1, y1)" patterns in free text.\n'
            'Use parse(..., output="pixel") for pixel coords (default), or\n'
            'parse(..., output="norm") for [0, 1] normalized coords.\n'
        )
        logger.info(msg)
        cls._has_logged_hint = True

    @classmethod
    def parse(
        cls,
        text: str,
        width: int,
        height: int,
        output: str = 'pixel',
        coord_scale: str = 'auto',
    ) -> np.ndarray:
        """
        Main entry.

        Args:
            text: raw model output.
            width, height: image size.
            output: 'pixel' for pixel coords, 'norm' for [0, 1] normalized coords.
            coord_scale: how to interpret model-output magnitudes.
              'auto'   — magnitude heuristic (≤1.5 → 0-1 float; ≤1000 → 0-1000
                         normalized; else → pixels). Default; correct for cr,
                         qwen3, gemini, gemma4 — every model that uses 0-1000
                         normalized output or 0-1 float.
              'pixel'  — values are absolute pixels; divide by image w/h. Set
                         explicitly by callers whose model emits pixel-scale
                         coordinates (e.g. MiMo / Qwen2.5-VL family).
              'normalized' — values are 0-1000 normalized; divide by 1000.
                         Available for callers that want to skip auto-detect.

        Returns:
            np.ndarray[N, 2]
        """
        if output not in ('pixel', 'norm'):
            raise ValueError(f'Point2DParser.parse: unsupported output={output}')
        if coord_scale not in ('auto', 'pixel', 'normalized'):
            raise ValueError(f'Point2DParser.parse: unsupported coord_scale={coord_scale}')

        pts = cls._json2pts(text, width, height, output=output, coord_scale=coord_scale)
        if pts is not None:
            return pts
        return cls._text2pts(text, width, height, output=output, coord_scale=coord_scale)

    @classmethod
    def _json2pts(
        cls,
        text: str,
        width: int = 640,
        height: int = 480,
        output: str = 'pixel',
        coord_scale: str = 'auto',
    ) -> np.ndarray | None:
        """
        Parse JSON/Python literals like:
        [
            {"point_2d": [x, y], "label": "..."},
            ...
        ]

        point_2d / point can be:
          - [0, 1] normalized
          - [0, 1000] normalized
          - pixels
        coord_scale: see Point2DParser.parse.
        """
        s = cls._strip_md_fence(text).strip()

        obj = None
        try:
            obj = json.loads(s)
        except Exception:
            pass

        if obj is None:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                return None

        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list):
            return None

        pts_norm = []  # Store uniformly as 0~1 coordinates
        w = float(width) if width else 1.0
        h = float(height) if height else 1.0

        for item in obj:
            # Support both dict format {"point_2d": [x, y]} and plain tuple/list [x, y].
            # Qwen/CR convention: key "point_2d", order [x, y]. Gemini/Gemma convention:
            # bare key "point" (no `_2d`), order [y, x] — swap to xy before normalize.
            swap_yx = False
            if isinstance(item, dict):
                if 'point_2d' in item:
                    pt = item.get('point_2d')
                elif 'point' in item:
                    pt = item.get('point')
                    swap_yx = True
                else:
                    pt = None
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pt = item
            else:
                continue
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                continue

            a, b = pt
            try:
                a = float(a)
                b = float(b)
            except Exception:
                continue
            # `float()` accepts NaN/Infinity; filter them here so they reach
            # neither the pixel rescale (where they'd cast to garbage ints)
            # nor the magnitude heuristic (where NaN comparisons are False
            # and the value falls through to the pixel branch).
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            x, y = (b, a) if swap_yx else (a, b)

            # map to [0,1] using the explicit coord_scale; 'auto' (default for
            # almost all callers) runs the magnitude heuristic.
            if coord_scale == 'pixel':
                x_norm = x / w
                y_norm = y / h
            elif coord_scale == 'normalized':
                x_norm = x / 1000.0
                y_norm = y / 1000.0
            else:  # 'auto' — magnitude heuristic
                max_abs = max(abs(x), abs(y))
                if 0.0 <= max_abs <= 1.5:
                    x_norm, y_norm = x, y
                elif 0.0 <= max_abs <= 1000.0:
                    x_norm = x / 1000.0
                    y_norm = y / 1000.0
                else:
                    # assume pixels
                    x_norm = x / w
                    y_norm = y / h

            pts_norm.append((x_norm, y_norm))

        if not pts_norm:
            return None

        pts_norm = np.array(pts_norm, dtype=float)

        if output == 'norm':
            return pts_norm

        # output == 'pixel'
        x_pix = np.clip(pts_norm[:, 0] * w, 0, w - 1)
        y_pix = np.clip(pts_norm[:, 1] * h, 0, h - 1)
        pts_pix = np.stack([x_pix, y_pix], axis=1).round().astype(int)
        return pts_pix

    @staticmethod
    def _text2pts(
        text: str,
        width: int = 640,
        height: int = 480,
        output: str = 'pixel',
        coord_scale: str = 'auto',
    ) -> np.ndarray:
        """
        Parse free-text patterns:
            (x, y) or (x0, y0, x1, y1)
        coord_scale: see Point2DParser.parse.
        """
        pattern = r'\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)'
        matches = re.findall(pattern, text)
        pts_norm = []
        w = float(width) if width else 1.0
        h = float(height) if height else 1.0

        for match in matches:
            nums = [float(num) for num in match.split(',')]
            max_abs = max(abs(v) for v in nums)
            is_norm = (0.0 <= max_abs <= 1.5)

            if len(nums) == 2:
                x, y = nums
                if coord_scale == 'pixel':
                    x_norm = x / w
                    y_norm = y / h
                elif coord_scale == 'normalized':
                    x_norm = x / 1000.0
                    y_norm = y / 1000.0
                else:  # 'auto' — magnitude heuristic
                    if is_norm:
                        x_norm, y_norm = x, y
                    elif max_abs <= 1000.0:
                        x_norm = x / 1000.0
                        y_norm = y / 1000.0
                    else:
                        x_norm = x / w
                        y_norm = y / h
                pts_norm.append((x_norm, y_norm))

            elif len(nums) == 4:
                x0, y0, x1, y1 = nums
                if is_norm:
                    x0 *= w
                    y0 *= h
                    x1 *= w
                    y1 *= h

                x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                x0_i, y0_i, x1_i, y1_i = map(int, map(round, (x0, y0, x1, y1)))
                h_box = max(0, y1_i - y0_i)
                w_box = max(0, x1_i - x0_i)
                if h_box > 0 and w_box > 0:
                    yy, xx = np.where(np.ones((h_box, w_box), dtype=np.uint8))
                    x_pix = xx + x0_i
                    y_pix = yy + y0_i
                    x_norm = x_pix / w
                    y_norm = y_pix / h
                    pts_norm.extend(zip(x_norm, y_norm))

        if not pts_norm:
            return np.empty((0, 2), dtype=float if output == 'norm' else int)

        pts_norm = np.array(pts_norm, dtype=float)

        if output == 'norm':
            return pts_norm

        # output == 'pixel'
        x_pix = np.clip(pts_norm[:, 0] * w, 0, w - 1)
        y_pix = np.clip(pts_norm[:, 1] * h, 0, h - 1)
        pts_pix = np.stack([x_pix, y_pix], axis=1).round().astype(int)
        return pts_pix

    @staticmethod
    def _strip_md_fence(text: str) -> str:
        s = text.strip()
        if not s.startswith('```'):
            return s

        first_nl = s.find('\n')
        if first_nl != -1:
            inner = s[first_nl + 1:]
        else:
            inner = s.lstrip('`')

        inner = inner.strip()
        if inner.endswith('```'):
            inner = inner[:-3]
        return inner.strip()
