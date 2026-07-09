import re

import regex  # Import regex module to support recursive matching of balanced brackets


class DefaultParser:
    def parse(self, response):
        if not response:
            return ""

        # Try to parse <|begin_of_box|>...</|end_of_box|> format - match from back to front
        box_pattern = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
        matches = re.findall(box_pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Take the last match

        # Then try to parse <answer></answer> format - match from back to front
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Take the last match

        # First try to parse \boxed{} format - match from back to front, use regex to support nested brackets
        boxed_pattern = r'\\boxed\{((?:[^{}]|\{[^}]*\})*)\}'
        matches = regex.findall(boxed_pattern, response)
        if matches:
            last_match = matches[-1]  # Take the last match
            # Find and remove all \text{} format in the last match
            text_pattern = r'\\text\{([^}]*)\}'
            text_matches = regex.findall(text_pattern, last_match)
            if text_matches:
                return text_matches[-1].strip()

            # 2. Handle truncated cases: ext{content} (missing \t)
            elif regex.search(r'ext\{[^}]*\}', last_match):
                ext_pattern = r'ext\{([^}]*)\}'
                ext_matches = regex.findall(ext_pattern, last_match)
                if ext_matches:
                    return ext_matches[-1].strip()

            # 3. Handle other possible text variants
            elif 'text{' in last_match:
                # Remove any form of text{...}
                cleaned = regex.sub(r'[\\]*text\{([^}]*)\}', r'\1', last_match)
                if cleaned.strip() != last_match.strip():
                    return cleaned.strip()

            # Try to parse \begin{array}...\end{array} format - use regex matching
            array_pattern = r'\\begin\{array\}((?:.|\n)*?)\\end\{array\}'
            array_matches = regex.findall(array_pattern, last_match)
            if array_matches:
                return array_matches[-1].strip()  # Take the last match

            # Try to parse \begin{bmatrix}...\end{bmatrix} format - use regex matching
            bmatrix_pattern = r'\\begin\{bmatrix\}((?:.|\n)*?)\\end\{bmatrix\}'
            bmatrix_matches = regex.findall(bmatrix_pattern, last_match)
            if bmatrix_matches:
                return bmatrix_matches[-1].strip()  # Take the last match
            return last_match.strip()

        # Finally try to parse Answer: format - match from back to front
        answer_matches = re.findall(r'Answer[:：]\s*(.*)', response, re.IGNORECASE | re.DOTALL)
        if answer_matches:
            return answer_matches[-1].strip()  # Take the last match

        return response  # return the original response


parser = {
    'default': DefaultParser().parse,
}
