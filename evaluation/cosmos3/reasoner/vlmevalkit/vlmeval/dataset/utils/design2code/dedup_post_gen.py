import difflib
import os
import re


def map_positions(clean_text, original_text):
    """Map positions from clean text back to original text."""
    map_clean_to_original = []
    original_idx = 0

    for clean_char in clean_text:
        while original_text[original_idx] != clean_char:
            original_idx += 1
        map_clean_to_original.append(original_idx)
        original_idx += 1

    return map_clean_to_original


def check_repetitive_content(
    file_path,
    chunk_size=100,
    repetition_threshold=5,
    similarity_threshold=0.8,
    debug=False,
):
    """
    Check repetitive content in a text file, ignoring HTML tags.

    It compares fixed-size chunks in cleaned text and maps the first repetitive
    position back to the original HTML string for optional truncation.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    content_no_html = re.sub("<.*?>", "", content)
    position_map = map_positions(content_no_html, content)

    chunks = [
        content_no_html[i:i + chunk_size]
        for i in range(0, len(content_no_html), chunk_size)
    ]

    seen = {}
    repetitive_start = len(content_no_html)
    for i, chunk in enumerate(chunks):
        for seen_chunk, indexes in seen.items():
            similarity = difflib.SequenceMatcher(None, chunk, seen_chunk).ratio()
            if similarity >= similarity_threshold:
                indexes.append(i)
                if len(indexes) >= repetition_threshold:
                    clean_start = min(repetitive_start, indexes[0] * chunk_size)
                    c_repetitive_start = (
                        position_map[clean_start]
                        if clean_start < len(position_map)
                        else len(content)
                    )
                    if c_repetitive_start < repetitive_start:
                        repetitive_start = c_repetitive_start
                break
        else:
            seen[chunk] = [i]

    repetitive = repetitive_start != len(content_no_html)
    start_position = repetitive_start

    if repetitive:
        print(f"[Warning] Repetitive content found in {file_path}, start at {start_position}")
        print(
            "[Warning] You might want to manually check whether the automatic repetition removal is correct."
        )
        if not debug:
            os.rename(file_path, file_path.replace(".html", "_old.txt"))
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content[:start_position])
        else:
            with open(file_path.replace(".html", "_new.html"), "w", encoding="utf-8") as file:
                file.write(content[:start_position])
