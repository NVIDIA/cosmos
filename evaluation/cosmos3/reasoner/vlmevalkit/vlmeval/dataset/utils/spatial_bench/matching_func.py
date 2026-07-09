import ast
import re

from num2words import num2words

# Zero-width characters (BOM, ZWSP, ZWNJ, ZWJ)
ZW_RE = re.compile(
    r'[\u200b\u200c\u200d\ufeff]'
)

# Generic numeric pattern: integer / float / scientific notation
NUMERIC_PATTERN = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'

_NUM_RE = re.compile(NUMERIC_PATTERN)

# <answer>...</answer> block with a leading option letter (A–J)
TAGGED_ANSWER_BLOCK = re.compile(
    r'<\s*answer\b[^>]*>'  # <answer ...>
    r'\s*'
    r'([A-Ja-j])'
    r'(?:\s*[\.．:：\)\]】、])?'  # Optional trailing punctuation, e.g. A. / A: / A) etc.
    r'.*?'
    r'<\s*/\s*answer\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)

# Numeric answer inside <answer>...</answer>
TAGGED_NUMERIC_ANSWER = re.compile(
    rf'<\s*answer\b[^>]*>'
    rf'\s*'
    rf'({NUMERIC_PATTERN})'
    rf'(?:[^\d<][^<]*)?'
    rf'<\s*/\s*answer\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)


# Matching func for mcq
def can_match_option(
    answer_text: str,
    choices=None,
    tail_lines: int = 6,
    tail_window: int = 800
):
    """
    Extract a single-choice option letter from free-form model output.

    Return:
        - 'A'..'F' (or restricted by `choices`) if a reliable match is found
        - False otherwise

    Procedure (1 → 7):
      1) Dynamic letter set: build allowed letters (default A–F; shrink if `choices` given)
      2) Block-level: <answer>...</answer> with leading letter (A–J allowed here)
      3) Tail anchors: a letter immediately before </answer> or </think> within the tail window
      4) Last-lines (after think-tail): scan last N lines after the last </think> (or <think>)
         - full-line single letter (markdown/brackets allowed)
         - labeled line start "A. ..."/"B) ..."
         - unique inline uppercase token not part of words
      5) Last-lines (global tail): same scan on the last N lines of the whole tail window
      6) Phrase-style conclusion (tail window only), e.g. "final answer: B"
         - capture is UPPERCASE only; use the last occurrence
      7) Global fallback (strict): after removing enumeration lines, accept ONLY one
         unique UPPERCASE standalone token across the entire text
    """
    # 1) Dynamic letter set
    if not isinstance(answer_text, str):
        return False
    text = ZW_RE.sub('', answer_text.strip())

    if choices:
        letters_sorted = ''.join(sorted({str(c).strip().upper()[:1] for c in choices if str(c)}))
        letters = ''.join([ch for ch in 'ABCDEFGHIJ' if ch in letters_sorted]) or 'ABCDEF'
    else:
        letters = 'ABCDEF'

    # 2) Block-level <answer>...</answer>
    m_block = TAGGED_ANSWER_BLOCK.search(text)
    if m_block:
        return m_block.group(1).upper()

    # 3) Tail anchors: before </answer> / </think>
    tail_block = text[-tail_window:]
    PAT_ANS = re.compile(
        rf'(?<![A-Za-z])'          # Not preceded by a letter
        rf'[\(\[\{{（【]?\s*'       # Optional left bracket + whitespace
        rf'([{letters}])\s*'       # One option letter from `letters`
        rf'[\)\]\}}）】]?\s*'       # Optional right bracket + whitespace
        rf'</\s*answer\s*>',       # Closing </answer> tag
        re.IGNORECASE,
    )
    PAT_THINK = re.compile(
        rf'(?<![A-Za-z])'
        rf'[\(\[\{{（【]?\s*'
        rf'([{letters}])\s*'
        rf'[\)\]\}}）】]?\s*'
        rf'</\s*think\s*>',
        re.IGNORECASE,
    )
    for pat in (PAT_ANS, PAT_THINK):
        m = pat.search(tail_block)
        if m:
            return m.group(1).upper()

    # Helpers for steps 4 & 5
    # Punctuation treated as tight boundary after a token (EN + CN)
    _PUNC_TIGHT = r"\.,:;!?\)\]】》」』，。；、：）】》」』"
    # Lines that start with "option A:" / "选项 A:" style prefixes
    OPTION_LINE_PREFIX = re.compile(
        r'^(?:[*_>\-\s]*)(?:option|选项)\s+[A-J]\s*[:：]',
        re.IGNORECASE,
    )
    # Lines whose content is a single option letter (with optional markdown/brackets)
    MD_SINGLE = re.compile(
        r'^\s*[*_`>（）\[\]【】\(\)]*\s*([A-Fa-f])\s*[*_`（）\[\]【】\(\)]*\s*$'
    )
    # Lines starting with "A. ...", "B) ...", etc.
    LINE_START_LABELED = re.compile(
        r'^\s*([A-F])\s*[\.．:：\)\]】、-]\s+',
        re.IGNORECASE,
    )
    # Inline standalone uppercase token (not part of a word), e.g. the A in "answer A."
    TOKEN_INLINE = re.compile(
        rf'(?<![A-Za-z])'          # Not preceded by a letter
        rf'[*_`（\[\{{\(]*\s*'     # optional prefix symbols + whitespace
        rf'([A-F])\s*'             # Capture a single uppercase letter A–F
        rf'[*_`）\]\}})]*'         # Optional suffix symbols
        rf'(?=$|[\s{_PUNC_TIGHT}])'  # Followed by end, whitespace, or tight punctuation
    )

    def _pick_from_lines(lines):
        for line in reversed([ln.strip() for ln in lines if ln.strip()]):
            if OPTION_LINE_PREFIX.search(line):
                continue
            m = MD_SINGLE.fullmatch(line)
            if m:
                return m.group(1).upper()
            m = LINE_START_LABELED.match(line)
            if m:
                return m.group(1).upper()
            tokens = [t.upper() for t in TOKEN_INLINE.findall(line)]
            if tokens:
                uniq = sorted(set(tokens))
                if len(uniq) == 1:
                    return uniq[0]
        return None

    # 4) Last-lines after think-tail
    if re.search(r'</\s*think\s*>', text, re.IGNORECASE):
        tail_segment = text[list(re.finditer(r'</\s*think\s*>', text, re.IGNORECASE))[-1].end():].strip()
    elif re.search(r'<\s*think\s*>', text, re.IGNORECASE):
        tail_segment = text[list(re.finditer(r'<\s*think\s*>', text, re.IGNORECASE))[-1].end():].strip()
    else:
        tail_segment = text

    pick = _pick_from_lines(tail_segment.splitlines()[-tail_lines:])
    if pick:
        return pick

    # 5) Last-lines in global tail window
    pick = _pick_from_lines(text[-tail_window:].splitlines()[-tail_lines:])
    if pick:
        return pick

    # 6) Phrase-style conclusion in tail window (last match)
    PHRASE_AFTER = re.compile(
        # Prefix phrases like "final answer", "the answer is", "答案", "我选", etc.
        rf'(?i)(?:final\s*answer|the\s*answer\s*is|answer(?:\s*is)?|correct\s*answer|'
        rf'答案|最终答案|结论|所以|因此|我选(?:择)?|选择|选)'
        rf'\s*[:：>＝=]?\s*'         # optional separator (:, ：, >, ＝, =)
        rf'[\(\[\{{（【]?\s*'        # optional left bracket
        rf'([{letters}])'           # option letter from `letters`
        rf'\s*[\)\]\}}）】]?'        # optional right bracket
        rf'(?:\b|[.)、。])'          # followed by boundary / end punctuation
    )

    m = PHRASE_AFTER.search(text)
    if m:
        return m.group(1).upper()

    # 7) Global fallback: uppercase-only & unique (skip enumerations)
    cleaned_lines = []
    for ln in text.splitlines():
        if OPTION_LINE_PREFIX.search(ln):
            continue
        cleaned_lines.append(ln)
    cleaned = "\n".join(cleaned_lines)

    TOKEN_UPPER_GLOBAL = re.compile(
        # Standalone option letter (not part of a word)
        rf'(?<![A-Za-z])'          # left side not a letter
        rf'[\(\[\{{（【]?\s*'       # optional left bracket + spaces
        rf'([{letters}])\s*'       # one letter from `letters`
        rf'[\)\]\}}）】]?'          # optional right bracket
        rf'(?![A-Za-z])'           # right side not a letter
    )
    tokens = TOKEN_UPPER_GLOBAL.findall(cleaned)

    uniq = sorted(set(tokens))
    if len(uniq) == 1:
        return uniq[0]

    return False


def _after_think(text: str) -> str:
    m_end = list(re.finditer(r'</\s*think\s*>', text, flags=re.IGNORECASE))
    if m_end:
        return text[m_end[-1].end():].strip()
    m_start = list(re.finditer(r'<\s*think\s*>', text, flags=re.IGNORECASE))
    if m_start:
        return text[m_start[-1].end():].strip()
    return text


def _last_number(s: str):
    nums = re.findall(_NUM_RE, s)
    if nums:
        return float(nums[-1])
    return None


def build_word2num(max_n: int = 99, lang: str = "en"):
    mapping = {}
    for i in range(0, max_n + 1):
        word = num2words(i, lang=lang)
        mapping[word] = i
    return mapping


WORD2NUM = build_word2num(20)
WORD_NUMBER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in WORD2NUM.keys()) + r")\b",
    flags=re.IGNORECASE,
)


def normalize_number_words(text: str) -> str:
    """
    Replace all recognizable English number phrases in `text`
    with their Arabic numeral strings.
    """
    def _repl(m: re.Match) -> str:
        key = m.group(1).lower()
        val = WORD2NUM.get(key)
        # Fallback: if not found in WORD2NUM, keep original text
        return str(val) if val is not None else m.group(0)

    return WORD_NUMBER_PATTERN.sub(_repl, text)


# Matching func for NA
def can_match_na(pred):
    try:
        if isinstance(pred, list):
            candidates = [str(pred[0])] if pred else []
        elif isinstance(pred, str) and pred.strip().startswith('[') and pred.strip().endswith(']'):
            seq = ast.literal_eval(pred)  # safer than eval
            candidates = [str(seq[0])] if isinstance(seq, list) and seq else [pred]
        else:
            candidates = [str(pred)]

        for raw in candidates:
            text = ZW_RE.sub('', raw.strip())
            text = normalize_number_words(text)

            # 1) <answer> ... </answer> numeric
            m = TAGGED_NUMERIC_ANSWER.search(text)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass

            # 2) after </think>: use *last* number in the tail
            tail = _after_think(text)
            v = _last_number(tail)
            if v is not None:
                return v

            # 3) global fallback:
            #    - if only one unique number -> that one
            #    - else last number in full text
            nums = re.findall(_NUM_RE, text)
            if nums:
                uniq = sorted(set(nums))
                if len(uniq) == 1:
                    return float(uniq[0])
                return float(nums[-1])

        return None
    except Exception:
        return None
