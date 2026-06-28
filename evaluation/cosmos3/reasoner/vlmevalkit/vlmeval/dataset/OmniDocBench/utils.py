import copy
import html
import json
import os
import pdb
import re
import shutil
import subprocess
import sys
import unicodedata
import uuid
from collections import defaultdict

import Levenshtein
import numpy as np
from bs4 import BeautifulSoup
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexencode import unicode_to_latex
from pylatexenc.latexwalker import (LatexCharsNode, LatexEnvironmentNode, LatexGroupNode,
                                    LatexMacroNode, LatexSpecialsNode, LatexWalker)
from scipy.optimize import linear_sum_assignment


def read_md_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    return content

def save_paired_result(preds, gts, save_path):
    save_result = []
    formula_id = 0
    for gt, pred in zip(gts, preds):
        save_result.append({
            "gt": gt,
            "pred": pred,
            "img_id": formula_id
        })
        formula_id += 1
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)

def remove_markdown_fences(content):
    content = re.sub(r'^```markdown\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'```\n?$', '', content, flags=re.MULTILINE)
    return content

# Standardize all consecutive characters
def replace_repeated_chars(input_str):
    input_str = re.sub(r'_{4,}', '____', input_str) # Replace more than 4 consecutive underscores with 4 underscores
    input_str = re.sub(r' {4,}', '    ', input_str)   # Replace more than 4 consecutive spaces with 4 spaces
    return re.sub(r'([^a-zA-Z0-9])\1{10,}', r'\1\1\1\1', input_str) # For other consecutive symbols (except numbers and letters), replace more than 10 occurrences with 4

# Special Unicode handling
def fullwidth_to_halfwidth(s):
    result = []
    for char in s:
        code = ord(char)
        # Convert full-width space to half-width space
        if code == 0x3000:
            code = 0x0020
        # Convert other full-width characters to half-width
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)

def find_special_unicode(s):
    special_chars = {}
    for char in s:
        if ord(char) > 127:  # Non-ASCII characters
            # unicode_name = unicodedata.name(char, None)
            unicode_name = unicodedata.category(char)
            special_chars[char] = f'U+{ord(char):04X} ({unicode_name})'
    return special_chars


inline_reg = re.compile(
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',
)

def textblock2unicode(text):
    inline_matches = inline_reg.finditer(text)
    removal_positions = []
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # print('-------- content-------', content)
        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        try:
            if any(char in clean_content for char in r'\^_'):
                if clean_content.endswith('\\'):
                    clean_content += ' '
                # inline_array.append(match.group(0))
                unicode_content = LatexNodes2Text().latex_to_text(clean_content)
                removal_positions.append((position[0], position[1], unicode_content))
        except:
            continue

    # Remove inline formulas from original text
    for start, end, unicode_content in sorted(removal_positions, reverse=True):
        text = text[:start] + unicode_content.strip() + text[end:]

    return text

def normalized_formula(text):
    # Normalize math formulas before matching
    filter_list = ['\\mathbf', '\\mathrm', '\\mathnormal', '\\mathit', '\\mathbb', '\\mathcal', '\\mathscr', '\\mathfrak', '\\mathsf', '\\mathtt',
                   '\\textbf', '\\text', '\\boldmath', '\\boldsymbol', '\\operatorname', '\\bm',
                   '\\symbfit', '\\mathbfcal', '\\symbf', '\\scriptscriptstyle', '\\notag',
                   '\\setlength', '\\coloneqq', '\\space', '\\thickspace', '\\thinspace', '\\medspace', '\\nobreakspace', '\\negmedspace',
                   '\\quad', '\\qquad', '\\enspace', '\\substackw', ' ']
                #    '\\left', '\\right', '{', '}', ' ']

    # delimiter_filter
    pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
    match = pattern.search(text)

    if match:
        text = match.group(1).strip()

    tag_pattern = re.compile(r"\\tag\{.*?\}")
    text = tag_pattern.sub('', text)
    hspace_pattern = re.compile(r"\\hspace\{.*?\}")
    text = hspace_pattern.sub('', text)
    begin_pattern = re.compile(r"\\begin\{.*?\}")
    text = begin_pattern.sub('', text)
    end_pattern = re.compile(r"\\end\{.*?\}")
    text = end_pattern.sub('', text)
    col_sep = re.compile(r"\\arraycolsep.*?\}")
    text = col_sep.sub('', text)
    text = text.strip('.')

    for filter_text in filter_list:
        text = text.replace(filter_text, '')

    # text = normalize_text(delimiter_filter(text))
    # text = delimiter_filter(text)
    text = text.lower()
    return text

def normalized_html_table(text):
    def process_table_html(md_i):
        """
        pred_md format edit
        """
        def process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()  # unwrap()会移除标签但保留其内容
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res=''
        table_res_no_space=''
        if '<table' in md_i.replace(" ","").replace("'",'"'):
            md_i = process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            # table_res = re.sub('<table.*?>','',table_res)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>',"",table_res)

            table_res = re.sub(r'\s+', " ", table_res)
            table_res_no_space = '<html><body><table border="1" >' + table_res.replace(' ','') + '</table></body></html>'
            # table_res_no_space = re.sub(' (style=".*?")',"",table_res_no_space)
            # table_res_no_space = re.sub(r'[ ]', " ", table_res_no_space)
            table_res_no_space = re.sub('colspan="', ' colspan="', table_res_no_space)
            table_res_no_space = re.sub('rowspan="', ' rowspan="', table_res_no_space)
            table_res_no_space = re.sub('border="', ' border="', table_res_no_space)

            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'
            # table_flow.append(table_res)
            # table_flow_no_space.append(table_res_no_space)

        return table_res, table_res_no_space

    def clean_table(input_str,flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>','',input_str)
        return input_str

    norm_text, _ = process_table_html(text)
    norm_text = clean_table(norm_text)
    return norm_text

def normalized_latex_table(text):
    def latex_template(latex_code):
        template = r'''
        \documentclass[border=20pt]{article}
        \usepackage{subcaption}
        \usepackage{url}
        \usepackage{graphicx}
        \usepackage{caption}
        \usepackage{multirow}
        \usepackage{booktabs}
        \usepackage{color}
        \usepackage{colortbl}
        \usepackage{xcolor,soul,framed}
        \usepackage{fontspec}
        \usepackage{amsmath,amssymb,mathtools,bm,mathrsfs,textcomp}
        \setlength{\parindent}{0pt}''' + \
        r'''
        \begin{document}
        ''' + \
        latex_code + \
        r'''
        \end{document}'''

        return template

    def process_table_latex(latex_code):
        SPECIAL_STRINGS= [
            ['\\\\vspace\\{.*?\\}', ''],
            ['\\\\hspace\\{.*?\\}', ''],
            ['\\\\rule\{.*?\\}\\{.*?\\}', ''],
            ['\\\\addlinespace\\[.*?\\]', ''],
            ['\\\\addlinespace', ''],
            ['\\\\renewcommand\\{\\\\arraystretch\\}\\{.*?\\}', ''],
            ['\\\\arraystretch\\{.*?\\}', ''],
            ['\\\\(row|column)?colors?\\{[^}]*\\}(\\{[^}]*\\}){0,2}', ''],
            ['\\\\color\\{.*?\\}', ''],
            ['\\\\textcolor\\{.*?\\}', ''],
            ['\\\\rowcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\columncolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\cellcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\colorbox\\{.*?\\}', ''],
            ['\\\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)', ''],
            [r'\s+', ' '],
            ['\\\\centering', ''],
            ['\\\\begin\\{table\\}\\[.*?\\]', '\\\\begin{table}'],
            ['\t', ''],
            ['@{}', ''],
            ['\\\\toprule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\bottomrule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\midrule(\\[.*?\\])?', '\\\\hline'],
            ['p\\{[^}]*\\}', 'l'],
            ['m\\{[^}]*\\}', 'c'],
            ['\\\\scalebox\\{[^}]*\\}\\{([^}]*)\\}', '\\1'],
            ['\\\\textbf\\{([^}]*)\\}', '\\1'],
            ['\\\\textit\\{([^}]*)\\}', '\\1'],
            ['\\\\cmidrule(\\[.*?\\])?\\(.*?\\)\\{([0-9]-[0-9])\\}', '\\\\cline{\\2}'],
            ['\\\\hline', ''],
            [r'\\multicolumn\{1\}\{[^}]*\}\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', r'\1']
        ]
        pattern = r'\\begin\{tabular\}.*\\end\{tabular\}'  # 注意这里不用 .*?
        matches = re.findall(pattern, latex_code, re.DOTALL)
        latex_code = ' '.join(matches)

        for special_str in SPECIAL_STRINGS:
            latex_code = re.sub(fr'{special_str[0]}', fr'{special_str[1]}', latex_code)

        return latex_code

    def convert_latex_to_html(latex_content, cache_dir='./temp'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        uuid_str = str(uuid.uuid1())
        with open(f'{cache_dir}/{uuid_str}.tex', 'w') as f:
            f.write(latex_template(latex_content))

        cmd = ['latexmlc', '--quiet', '--nocomments', f'--log={cache_dir}/{uuid_str}.log',
               f'{cache_dir}/{uuid_str}.tex', f'--dest={cache_dir}/{uuid_str}.html']
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(f'{cache_dir}/{uuid_str}.html', 'r') as f:
                html_content = f.read()

            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            tables = [f'<table>{table}</table>' for table in tables]
            html_content = '\n'.join(tables)

        except Exception as e:
            html_content = ''

        shutil.rmtree(cache_dir)
        return html_content

    html_text = convert_latex_to_html(text)
    normlized_tables = normalized_html_table(html_text)
    return normlized_tables


def normalized_table(text, format='html'):
    if format not in ['html', 'latex']:
        raise ValueError('Invalid format: {}'.format(format))
    else:
        return globals()['normalized_{}_table'.format(format)](text)


def textblock_with_norm_formula(text):
    inline_matches = inline_reg.finditer(text)
    removal_positions = []
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # print('-------- content-------', content)

        norm_content = normalized_formula(content)
        removal_positions.append((position[0], position[1], norm_content))

    # Remove inline formulas from original text
    for start, end, norm_content in sorted(removal_positions, reverse=True):
        text = text[:start] + norm_content.strip() + text[end:]

    return text


def inline_filter_unicode(text):
    # Ensure text is string type
    if not isinstance(text, str):
        text = str(text)

    # Replace inline formula boundary markers
    #print('--------text-------',text)
    placeholder = '__INLINE_FORMULA_BOUNDARY__'
    text_copy = text.replace('$', placeholder).replace('\\(', placeholder).replace('\\)', placeholder)
    #print('--------text_copy-------',text_copy)
    # Convert LaTeX content to Unicode representation
    text_copy = LatexNodes2Text().latex_to_text(text_copy)
    #print('--------text_copy---unicode----',text_copy)
    # Restore boundary markers
    text_copy = text_copy.replace(placeholder, '$')

    inline_array = []
    inline_matches = inline_reg.finditer(text_copy)
    # Record positions of inline formulas to be removed
    removal_positions = []

    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        print('-------- content-------', content)
        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        if any(char in clean_content for char in r'\^_'):
            # inline_array.append(match.group(0))
            inline_array.append({
                'category_type': 'equation_inline',
                'position': position,
                'content': content,
            })
            removal_positions.append((position[0], position[1]))

    # Remove inline formulas from original text
    for start, end in sorted(removal_positions, reverse=True):
        text = text[:start] + text[end:]

    return text, inline_array

def inline_filter(text):
    # Ensure text is string type
    if not isinstance(text, str):
        text = str(text)

    inline_array = []
    inline_matches = inline_reg.finditer(text)

    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # print('inline_content: ', content)

        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        if any(char in clean_content for char in r'\^_'):
            # inline_array.append(match.group(0))
            inline_array.append({
                'category_type': 'equation_inline',
                'position': position,
                'content': match.group(0),
            })
            text = text.replace(match.group(0), '')
            # print('-----Found inline formula: ', match.group(0))
        else:
            text = text.replace(match.group(0), content)

    return text, inline_array

# Text OCR quality check processing:
def clean_string(input_string):
    # Use regex to keep Chinese characters, English letters and numbers
    input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    cleaned_string = re.sub(r'[^\w\u4e00-\u9fff]', '', input_string)
    return cleaned_string

def extract_tabular(text):
    begin_pattern = r'\\begin{tabular}'
    end_pattern = r'\\end{tabular}'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []

    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])

        if not begin_match and not end_match:
            break

        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1

    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_tabular(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions

# math reg
    # r'\\begin{equation\*?}(.*?)\\end{equation\*?}|'
    # r'\\begin{align\*?}(.*?)\\end{align\*?}|'
    # r'\\begin{gather\*?}(.*?)\\end{gather\*?}|'
display_reg = re.compile(
    r'\$\$(.*?)\$\$|'
    r'\\\[(.*?)\\\]|'
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',
    re.DOTALL
)

# inline_reg = re.compile(
#     r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)|'
#     r'\\\((.*?)\\\)',
# )
inline_reg = re.compile(
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',
)

# table
table_reg = re.compile(
    r'\\begin{table\*?}(.*?)\\end{table\*?}|'
    r'\\begin{tabular\*?}(.*?)\\end{tabular\*?}',
    re.DOTALL
)
md_table_reg = re.compile(
    r'\|\s*.*?\s*\|\n',
    re.DOTALL)
html_table_reg = re.compile(
    r'(<table.*?</table>)',
    re.DOTALL
)

# title
title_reg = re.compile(
    r'^\s*#.*$',
    re.MULTILINE)

# img
img_pattern = r'!\[.*?\]\(.*?\)'

# code block
code_block_reg = re.compile(
    r'```(\w+)\n(.*?)```',
    re.DOTALL
)


def md_tex_filter(content):
    '''
    Input: 1 page md or tex content - String
    Output: text, display, inline, table, title, code - list
    '''
    content = re.sub(img_pattern, '', content)  # remove image
    content = remove_markdown_fences(content)   # remove markdown fences
    content = replace_repeated_chars(content) # replace all consecutive characters



    pred_all = []
    latex_table_array, table_positions = extract_tex_table(content)
    for latex_table, position in zip(latex_table_array, table_positions):
        position = [position[0], position[0]+len(latex_table)]   # !!!
        pred_all.append({
            'category_type': 'latex_table',
            'position': position,
            'content': latex_table
        })
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace latex table with space


    # extract html table
    html_table_array, table_positions = extract_html_table(content)
    for html_table, position in zip(html_table_array, table_positions):
        position = [position[0], position[0]+len(html_table)]
        pred_all.append({
            'category_type': 'html_table',
            'position': position,
            'content': html_table
        })
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace html table with space

    # extract interline formula
    display_matches = display_reg.finditer(content)
    for match in display_matches:
        matched = match.group(0)
        if matched:
            single_line = ''.join(matched.split())
            position = [match.start(), match.end()]
            # replace $$ with \[\]
            dollar_pattern = re.compile(r'\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)', re.DOTALL)
            sub_match = dollar_pattern.search(single_line)
            if sub_match is None:
                # pass
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line
                })
            elif sub_match.group(1):
                single_line = re.sub(dollar_pattern, r'\\[\1\\]', single_line)
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace equation with space
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line
                })
            else:
                single_line = re.sub(dollar_pattern, r'\\[\2\3\\]', single_line)
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line,
                    'fine_category_type': 'equation_inline'
                })


    # extract md table with ||
    md_table_mathces = md_table_reg.findall(content+'\n')
    if len(md_table_mathces) >= 2:
        # print("md table found!")
        # print("content:", content)
        content = convert_markdown_to_html(content)
        # print('----------content after converting md table to html:', content)
        html_table_matches = html_table_reg.finditer(content)
        if html_table_matches:
            for match in html_table_matches:
                matched = match.group(0)
                position = [match.start(), match.end()]
                # content = content.replace(match, '')
                # print('content after removing the md table:', content)
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace md table with space
                pred_all.append({
                    'category_type': 'html_table',
                    'position': position,
                    'content': matched.strip(),
                    'fine_category_type': 'md2html_table'
                })
    # print('---------After md table: \n', content)

    # extract code blocks
    code_matches = code_block_reg.finditer(content)
    if code_matches:
        for match in code_matches:
            position = [match.start(), match.end()]
            language = match.group(1)
            code = match.group(2).strip()
            # content = content.replace(match.group(0), '')
            content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace code block with space
            pred_all.append({
                'category_type': 'text_all',
                'position': position,
                'content': code,
                'language': language,
                'fine_category_type': 'code'
            })


    # Remove latex style
    content = re.sub(r'\\title\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\title\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\section\*?\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\section\*?\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)

    # extract texts
    res = content.split('\n\n')
    if len(res) == 1:
        res = content.split('\n')  # some models do not use double newlines, so use single newlines to split

    content_position = 0
    for text in res:
        position = [content_position, content_position+len(text)]
        content_position += len(text)
        text = text.strip()
        text = text.strip('\n')
        # print('ori_text: ', text)
        text = '\n'.join([_.strip() for _ in text.split('\n') if _.strip()])   # avoid some single newline content with many spaces
        # print('after strip text: ', text)

        if text:  # Check if the stripped text is not empty
            if text.startswith('<table') and text.endswith('</table>'):
                pred_all.append({
                    'category_type': 'html_table',
                    'position': position,
                    'content': text,
                })

            elif text.startswith('$') and text.endswith('$'):
                if text.replace('$', '').strip():
                    pred_all.append({
                        'category_type': 'equation_isolated',
                        'position': position,
                        'content': text.strip(),
                    })
            else:
                text = text.strip()
                if text:
                    pred_all.append({
                        'category_type': 'text_all',
                        'position': position,
                        'content': text,
                        'fine_category_type': 'text_block'
                    })

    pred_dataset = defaultdict(list)
    pred_all = sorted(pred_all, key=lambda x: x['position'][0])
    for item in pred_all:
        pred_dataset[item['category_type']].append(item)
    # pdb.set_trace()
    return pred_dataset


def extract_tex_table(content):
    tables = []
    tables_positions = []

    pattern = r'\\begin{table}(.*?)\\end{table}'
    for match in re.finditer(pattern, content, re.DOTALL):
        start_pos = match.start()
        end_pos = match.end()
        table_content = match.group(0)
        tables.append(table_content)
        tables_positions.append((start_pos, end_pos))
        content = content[:start_pos] + ' '*(end_pos-start_pos) + content[end_pos:]

    tabulars, tabular_positions = extract_tabular(content)
    all_tables = tables + tabulars
    all_positions = tables_positions + tabular_positions

    all_result = sorted([[pos, table]for pos, table in zip(all_positions, all_tables)], key=lambda x: x[0][0])
    all_tables = [x[1] for x in all_result]
    all_positions = [x[0] for x in all_result]

    return all_tables, all_positions


def extract_html_table(text):
    begin_pattern = r'<table(?:[^>]*)>'
    end_pattern = r'</table>'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []

    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])

        if not begin_match and not end_match:
            break

        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1

    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_html_table(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions


def extract_node_content(node):
    """ Recursively extract content from LatexEnvironmentNode and rebuild LaTeX table representation """
    if isinstance(node, LatexCharsNode):
        return node.chars  # Use chars attribute
    elif isinstance(node, LatexGroupNode):
        return "{" + "".join(extract_node_content(n) for n in node.nodelist) + "}"
    elif isinstance(node, LatexMacroNode):
        # Extract macro command and its arguments
        macro_content = "\\" + node.macroname
        if node.nodeargs:
            macro_content += "".join([extract_node_content(arg) for arg in node.nodeargs])
        return macro_content
    elif isinstance(node, LatexEnvironmentNode):
        # Extract environment, preserve environment name and arguments
        content = "\\begin{" + node.environmentname + "}"
        if node.nodeargd and node.nodeargd.argnlist:
            # content += "".join("{" + extract_node_content(arg) + "}" for arg in node.nodeargd)
            # content += "".join("{" + extract_node_content(node.nodeargd) + "}")
            content += "{" + extract_node_content(node.nodeargd.argnlist[0]) + "}"
        if node.nodelist:
            content += "".join(extract_node_content(n) for n in node.nodelist)
        content += "\\end{" + node.environmentname + "}"
        return content
    elif isinstance(node, LatexSpecialsNode):  # Changed to LatexSpecialsNode
        return node.specials_chars
    else:
        return ""

def get_node_end_pos(node):
    """Recursively determine the end position of a node"""
    if hasattr(node, 'nodelist') and node.nodelist:
        # If the node has child nodes, recursively find the end position of the last child node
        return get_node_end_pos(node.nodelist[-1])
    elif hasattr(node, 'pos_end'):
        # If the node has pos_end attribute, return it directly
        return node.pos_end
    else:
        # If there are no child nodes, assume the node ends at the last character of its content
        return node.pos + len(str(node))

def remove_tex_table(content):
    tables, positions = extract_tex_table(content)

    # Delete in reverse order by position to avoid affecting unprocessed start positions
    for start, end in sorted(positions, reverse=True):
        content = content[:start] + content[end:]  # Remove table content

    return content



def get_pred_category_type(pred_idx, pred_items):
    # if pred_idx:
    if pred_items[pred_idx].get('fine_category_type'):
        pred_pred_category_type = pred_items[pred_idx]['fine_category_type']
    else:
        pred_pred_category_type = pred_items[pred_idx]['category_type']
    # else:
    #     pred_pred_category_type = ""
    return pred_pred_category_type


def compute_edit_distance_matrix_new(gt_lines, matched_lines):
    try:
        distance_matrix = np.zeros((len(gt_lines), len(matched_lines)))
        for i, gt_line in enumerate(gt_lines):
            for j, matched_line in enumerate(matched_lines):
                if len(gt_line) == 0 and len(matched_line) == 0:
                    distance_matrix[i][j] = 0
                else:
                    distance_matrix[i][j] = Levenshtein.distance(gt_line, matched_line) / max(len(matched_line), len(gt_line))
        return distance_matrix
    except ZeroDivisionError:
        #print("ZeroDivisionError occurred. Outputting norm_gt_lines and norm_pred_lines:")
        # print("norm_gt_lines:", gt_lines)
        # print("norm_pred_lines:", matched_lines)
        raise

def get_gt_pred_lines(gt_items, pred_items, line_type):
    norm_html_lines = []
    gt_lines = []
    gt_cat_list = []
    for item in gt_items:
        if item.get('fine_category_type'):
            gt_cat_list.append(item['fine_category_type'])
        else:
            gt_cat_list.append(item['category_type'])
        if item.get('content'):
            gt_lines.append(str(item['content']))
            norm_html_lines.append(str(item['content']))
        elif line_type == 'text':
            gt_lines.append(str(item['text']))
        elif line_type == 'html_table':
            gt_lines.append(str(item['html']))
        elif line_type == 'formula':
            gt_lines.append(str(item['latex']))
        elif line_type == 'latex_table':
            gt_lines.append(str(item['latex']))
            norm_html_lines.append(str(item['html']))

    pred_lines = [str(item['content']) for item in pred_items]


    if line_type == 'formula':
        norm_gt_lines = [normalized_formula(_) for _ in gt_lines]
        norm_pred_lines = [normalized_formula(_) for _ in pred_lines]
    elif line_type == 'text':
        # norm_gt_lines = [textblock_with_norm_formula(_) for _ in gt_lines]
        # norm_pred_lines = [textblock_with_norm_formula(_) for _ in pred_lines]
        norm_gt_lines = [clean_string(textblock2unicode(_)) for _ in gt_lines]
        norm_pred_lines = [clean_string(textblock2unicode(_)) for _ in pred_lines]
        # norm_gt_lines = get_norm_text_lines(gt_lines)
        # norm_pred_lines = get_norm_text_lines(pred_lines)
    else:
        norm_gt_lines = gt_lines
        norm_pred_lines = pred_lines

    if line_type == 'latex_table':
        gt_lines = norm_html_lines


    filtered_lists = [(a, b, c) for a, b, c in zip(gt_lines, norm_gt_lines, gt_cat_list) if a and b]

    # decompress to three lists
    if filtered_lists:
        gt_lines_c, norm_gt_lines_c, gt_cat_list_c = zip(*filtered_lists)

        # convert to lists
        gt_lines_c = list(gt_lines_c)
        norm_gt_lines_c = list(norm_gt_lines_c)
        gt_cat_list_c = list(gt_cat_list_c)
    else:
        gt_lines_c = []
        norm_gt_lines_c = []
        gt_cat_list_c = []

    # pred's empty values
    filtered_lists = [(a, b) for a, b in zip(pred_lines, norm_pred_lines) if a and b]

    # decompress to two lists
    if filtered_lists:
        pred_lines_c, norm_pred_lines_c = zip(*filtered_lists)

        # convert to lists
        pred_lines_c = list(pred_lines_c)
        norm_pred_lines_c = list(norm_pred_lines_c)
    else:
        pred_lines_c = []
        norm_pred_lines_c = []

    return gt_lines_c, norm_gt_lines_c, gt_cat_list_c, pred_lines_c, norm_pred_lines_c
    # return gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines


def match_gt2pred_simple(gt_items, pred_items, line_type, img_name):

    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines = get_gt_pred_lines(gt_items, pred_items, line_type)

    match_list = []
    if not norm_gt_lines: # not matched pred should be concatenated
        # print("One of the lists is empty. Returning an empty gt result.")
        # for pred_idx in range(len(norm_pred_lines)):
        pred_idx_list = range(len(norm_pred_lines))
        match_list.append({
            'gt_idx': [""],
            'gt': "",
            'pred_idx': pred_idx_list,
            'pred': ''.join(pred_lines[_] for _ in pred_idx_list),
            'gt_position': [""],
            'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
            'norm_gt': "",
            'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
            'gt_category_type': "",
            'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
            'gt_attribute': [{}],
            'edit': 1,
            'img_id': img_name
        })
        return match_list
    elif not norm_pred_lines: # not matched gt should be separated
        # print("One of the lists is empty. Returning an empty pred result.")
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
                'pred_position': "",
                'norm_gt': norm_gt_lines[gt_idx],
                'norm_pred': "",
                'gt_category_type': gt_cat_list[gt_idx],
                'pred_category_type': "",
                'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
                'edit': 1,
                'img_id': img_name
            })
        return match_list

    cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, norm_pred_lines)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)


    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx in row_ind:
            row_i = list(row_ind).index(gt_idx)
            pred_idx = int(col_ind[row_i])
            pred_line = pred_lines[pred_idx]
            norm_pred_line = norm_pred_lines[pred_idx]
            edit = cost_matrix[gt_idx][pred_idx]
            # print('edit_dist', edit)
            # if edit > 0.7:
            #     print('! Not match')
        else:
            # print('No match pred')
            pred_idx = ""
            pred_line = ""
            norm_pred_line = ""
            edit = 1

        match_list.append({
            'gt_idx': [gt_idx],
            'gt': gt_lines[gt_idx],
            'norm_gt': norm_gt_lines[gt_idx],
            'gt_category_type': gt_cat_list[gt_idx],
            'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
            'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
            'pred_idx': [pred_idx],
            'pred': pred_line,
            'norm_pred': norm_pred_line,
            'pred_category_type': get_pred_category_type(pred_idx, pred_items) if pred_idx else "",
            'pred_position': pred_items[pred_idx]['position'][0] if pred_idx else "",
            'edit': edit,
            'img_id': img_name
        })
        # print('-'*10)
        # [([0,1], 0),(2, 1), (1,2)] --> [0,2,1]/[0,1,2]

    pred_idx_list = [pred_idx for pred_idx in range(len(norm_pred_lines)) if pred_idx not in col_ind] # get not matched preds
    if pred_idx_list: # if there are still remaining pred_idx, concatenate all preds
        match_list.append({
            'gt_idx': [""],
            'gt': "",
            'pred_idx': pred_idx_list,
            'pred': ''.join(pred_lines[_] for _ in pred_idx_list),
            'gt_position': [""],
            'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
            'norm_gt': "",
            'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
            'gt_category_type': "",
            'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
            'gt_attribute': [{}],
            'edit': 1,
            'img_id': img_name
        })
    return match_list


def match_gt2pred_no_split(gt_items, pred_items, line_type, img_name):
    # directly concatenate gt and pred by position
    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines = get_gt_pred_lines(gt_items, pred_items, line_type)
    gt_line_with_position = []
    for gt_line, norm_gt_line, gt_item in zip(gt_lines, norm_gt_lines, gt_items):
        gt_position = gt_item['order'] if gt_item.get('order') else gt_item.get('position', [""])[0]
        if gt_position:
            gt_line_with_position.append((gt_position, gt_line, norm_gt_line))
    sorted_gt_lines = sorted(gt_line_with_position, key=lambda x: x[0])
    gt = '\n\n'.join([_[1] for _ in sorted_gt_lines])
    norm_gt = '\n\n'.join([_[2] for _ in sorted_gt_lines])
    pred_line_with_position = [(pred_item['position'], pred_line, pred_norm_line) for pred_line, pred_norm_line, pred_item in zip(pred_lines, norm_pred_lines, pred_items)]
    sorted_pred_lines = sorted(pred_line_with_position, key=lambda x: x[0])
    pred = '\n\n'.join([_[1] for _ in sorted_pred_lines])
    norm_pred = '\n\n'.join([_[2] for _ in sorted_pred_lines])
    # edit = Levenshtein.distance(norm_gt, norm_pred)/max(len(norm_gt), len(norm_pred))
    if norm_gt or norm_pred:
        return [{
                'gt_idx': [0],
                'gt': gt,
                'norm_gt': norm_gt,
                'gt_category_type': "text_merge",
                'gt_position': [""],
                'gt_attribute': [{}],
                'pred_idx': [0],
                'pred': pred,
                'norm_pred': norm_pred,
                'pred_category_type': "text_merge",
                'pred_position': "",
                # 'edit': edit,
                'img_id': img_name
            }]
    else:
        return []


import copy
import pdb
from collections import Counter, defaultdict

import evaluate
# from rapidfuzz.distance import Levenshtein
import Levenshtein
import numpy as np
from Levenshtein import distance as Levenshtein_distance
from scipy.optimize import linear_sum_assignment


def match_gt2pred_quick(gt_items, pred_items, line_type, img_name):

    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines= get_gt_pred_lines(gt_items, pred_items, line_type)
    all_gt_indices = set(range(len(norm_gt_lines)))
    all_pred_indices = set(range(len(norm_pred_lines)))

    if not norm_gt_lines:
        match_list = []
        for pred_idx in range(len(norm_pred_lines)):
            match_list.append({
                'gt_idx': [""],
                'gt': "",
                'pred_idx': [pred_idx],
                'pred': pred_lines[pred_idx],
                'gt_position': "",
                'pred_position': pred_items[pred_idx]['position'][0],
                'norm_gt': "",
                'norm_pred': norm_pred_lines[pred_idx],
                'gt_category_type': "",
                'pred_category_type': get_pred_category_type(pred_idx, pred_items),
                'gt_attribute': [{}],
                'edit': 1,
                'img_id': img_name
            })
        return match_list
    elif not norm_pred_lines:
        match_list = []
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
                'pred_position': "",
                'norm_gt': norm_gt_lines[gt_idx],
                'norm_pred': "",
                'gt_category_type': gt_cat_list[gt_idx],
                'pred_category_type': "",
                'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
                'edit': 1,
                'img_id': img_name
            })
        return match_list
    elif len(norm_gt_lines) == 1 and len(norm_pred_lines) == 1:
        edit_distance = Levenshtein_distance(norm_gt_lines[0], norm_pred_lines[0])
        normalized_edit_distance = edit_distance / max(len(norm_gt_lines[0]), len(norm_pred_lines[0]))
        return [{
            'gt_idx': [0],
            'gt': gt_lines[0],
            'pred_idx': [0],
            'pred': pred_lines[0],
            'gt_position': [gt_items[0].get('order') if gt_items[0].get('order') else gt_items[0].get('position', [""])[0]],
            'pred_position': pred_items[0]['position'][0],
            'norm_gt': norm_gt_lines[0],
            'norm_pred': norm_pred_lines[0],
            'gt_category_type': gt_cat_list[0],
            'pred_category_type': get_pred_category_type(0, pred_items),
            'gt_attribute': [gt_items[0].get("attribute", {})],
            'edit': normalized_edit_distance,
            'img_id': img_name
        }]

    cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, norm_pred_lines)

    matched_col_idx, row_ind, cost_list = cal_final_match(cost_matrix, norm_gt_lines, norm_pred_lines)

    gt_lens_dict, pred_lens_dict = initialize_indices(norm_gt_lines, norm_pred_lines)

    matches, unmatched_gt_indices, unmatched_pred_indices = process_matches(matched_col_idx, row_ind, cost_list, norm_gt_lines, norm_pred_lines, pred_lines)

    matching_dict = fuzzy_match_unmatched_items(unmatched_gt_indices, norm_gt_lines, norm_pred_lines)

    final_matches = merge_matches(matches, matching_dict)

    recalculate_edit_distances(final_matches, gt_lens_dict, norm_gt_lines, norm_pred_lines)

    converted_results = convert_final_matches(final_matches, norm_gt_lines, norm_pred_lines)

    merged_results = merge_duplicates_add_unmatched(converted_results, norm_gt_lines, norm_pred_lines, gt_lines, pred_lines, all_gt_indices, all_pred_indices)

    for entry in merged_results:
            entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
            entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
            entry['gt_position'] = [gt_items[_].get('order') if gt_items[_].get('order') else gt_items[_].get('position', [""])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [""]
            entry['pred_position'] = pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [""] else ""
            entry['gt'] = ''.join([gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            entry['pred'] = ''.join([pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""
            entry['norm_gt'] = ''.join([norm_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            entry['norm_pred'] = ''.join([norm_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

            if entry['gt_idx'] != [""]:
                ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
                gt_cagegory_clean = [gt_cat_list[_] for _ in entry['gt_idx'] if gt_cat_list[_] not in ignore_type]
                if gt_cagegory_clean:
                    entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0]
                else:
                    entry['gt_category_type'] = Counter([gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0]
            else:
                entry['gt_category_type'] = ""
            entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], pred_items) if entry['pred_idx'] != [""] else ""
            entry['gt_attribute'] = [gt_items[_].get("attribute", {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [{}]
            entry['img_id'] = img_name

    return merged_results


def merge_duplicates_add_unmatched(converted_results, norm_gt_lines, norm_pred_lines, gt_lines, pred_lines, all_gt_indices, all_pred_indices):
    merged_results = []
    processed_pred = set()
    processed_gt = set()

    for entry in converted_results:
        pred_idx = tuple(entry['pred_idx']) if isinstance(entry['pred_idx'], list) else (entry['pred_idx'],)
        if pred_idx not in processed_pred and pred_idx != ("",):
            merged_entry = {
                'gt_idx': [entry['gt_idx']],
                'gt': entry['gt'],
                'pred_idx': entry['pred_idx'],
                'pred': entry['pred'],
                'edit': entry['edit']
            }
            for other_entry in converted_results:
                other_pred_idx = tuple(other_entry['pred_idx']) if isinstance(other_entry['pred_idx'], list) else (other_entry['pred_idx'],)
                if other_pred_idx == pred_idx and other_entry is not entry:
                    merged_entry['gt_idx'].append(other_entry['gt_idx'])
                    merged_entry['gt'] += other_entry['gt']
                    processed_gt.add(other_entry['gt_idx'])
            merged_results.append(merged_entry)
            processed_pred.add(pred_idx)
            processed_gt.add(entry['gt_idx'])

    for entry in converted_results:
        if entry['gt_idx'] not in processed_gt:
            merged_results.append(entry)

    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx not in processed_gt:
            merged_results.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'edit': 1
            })
    return merged_results




def formula_format(formula_matches, img_name):
    return [
        {
            "gt": item["gt"],
            "pred": item["pred"],
            "img_id": f"{img_name}_{i}"
        }
        for i, item in enumerate(formula_matches)
    ]


def merge_lists_with_sublists(main_list, sub_lists):
    main_list_final = list(copy.deepcopy(main_list))
    for sub_list in sub_lists:
        pop_idx = main_list_final.index(sub_list[0])
        for _ in sub_list:
            main_list_final.pop(pop_idx)
        main_list_final.insert(pop_idx, sub_list)
    return main_list_final


def sub_pred_fuzzy_matching(gt, pred):

    min_d = float('inf')
    # pos = -1

    gt_len = len(gt)
    pred_len = len(pred)

    if gt_len >= pred_len and pred_len > 0:
        for i in range(gt_len - pred_len + 1):
            sub = gt[i:i + pred_len]
            dist = Levenshtein_distance(sub, pred)/pred_len
            if dist < min_d:
                min_d = dist
                pos = i

        return min_d
    else:
        return False

def sub_gt_fuzzy_matching(pred, gt):

    min_d = float('inf')
    pos = ""
    matched_sub = ""
    gt_len = len(gt)
    pred_len = len(pred)

    if pred_len >= gt_len and gt_len > 0:
        for i in range(pred_len - gt_len + 1):
            sub = pred[i:i + gt_len]
            dist = Levenshtein.distance(sub, gt)  /gt_len
            if dist < min_d:
                min_d = dist
                pos = i
                matched_sub = sub
        return min_d, pos, gt_len, matched_sub
    else:
        return 1, "", gt_len, ""


def get_final_subset(subset_certain, subset_certain_cost):
    if not subset_certain or not subset_certain_cost:
        return []

    subset_turple = sorted([(a, b) for a, b in zip(subset_certain, subset_certain_cost)], key=lambda x: x[0][0])

    group_list = defaultdict(list)
    group_idx = 0
    group_list[group_idx].append(subset_turple[0])

    for item in subset_turple[1:]:
        overlap_flag = False
        for subset in group_list[group_idx]:
            for idx in item[0]:
                if idx in subset[0]:
                    overlap_flag = True
                    break
            if overlap_flag:
                break
        if overlap_flag:
            group_list[group_idx].append(item)
        else:
            group_idx += 1
            group_list[group_idx].append(item)

    final_subset = []
    for _, group in group_list.items():
        if len(group) == 1:
            final_subset.append(group[0][0])
        else:
            path_dict = defaultdict(list)
            path_idx = 0
            path_dict[path_idx].append(group[0])

            for subset in group[1:]:
                new_path = True
                for path_idx_s, path_items in path_dict.items():
                    is_dup = False
                    is_same = False
                    for path_item in path_items:
                        if path_item[0] == subset[0]:
                            is_dup = True
                            is_same = True
                            if path_item[1] > subset[1]:
                                path_dict[path_idx_s].pop(path_dict[path_idx_s].index(path_item))
                                path_dict[path_idx_s].append(subset)
                        else:
                            for num_1 in path_item[0]:
                                for num_2 in subset[0]:
                                    if num_1 == num_2:
                                        is_dup = True
                    if not is_dup:
                        path_dict[path_idx_s].append(subset)
                        new_path = False
                    if is_same:
                        new_path = False
                if new_path:
                    path_idx = len(path_dict.keys())
                    path_dict[path_idx].append(subset)

            saved_cost = float('inf')
            saved_subset = []
            for path_idx, path in path_dict.items():
                avg_cost = sum([i[1] for i in path]) / len(path)
                if avg_cost < saved_cost:
                    saved_subset = [i[0] for i in path]
                    saved_cost = avg_cost

            final_subset.extend(saved_subset)

    return final_subset

def judge_pred_merge(gt_list, pred_list, threshold=0.6):
    if len(pred_list) == 1:
        return False, False

    cur_pred = ' '.join(pred_list[:-1])
    merged_pred = ' '.join(pred_list)

    cur_dist = Levenshtein.distance(gt_list[0], cur_pred) / max(len(gt_list[0]), len(cur_pred))
    merged_dist = Levenshtein.distance(gt_list[0], merged_pred) / max(len(gt_list[0]), len(merged_pred))

    if merged_dist > cur_dist:
        return False, False

    cur_fuzzy_dists = [sub_pred_fuzzy_matching(gt_list[0], cur_pred) for cur_pred in pred_list[:-1]]
    if any(dist is False or dist > threshold for dist in cur_fuzzy_dists):
        return False, False

    add_fuzzy_dist = sub_pred_fuzzy_matching(gt_list[0], pred_list[-1])
    if add_fuzzy_dist is False:
        return False, False

    merged_pred_flag = add_fuzzy_dist < threshold
    continue_flag = len(merged_pred) <= len(gt_list[0])

    return merged_pred_flag, continue_flag

def deal_with_truncated(cost_matrix, norm_gt_lines, norm_pred_lines):
    matched_first = np.argwhere(cost_matrix < 0.25)
    masked_gt_idx = [i[0] for i in matched_first]
    unmasked_gt_idx = [i for i in range(cost_matrix.shape[0]) if i not in masked_gt_idx]
    masked_pred_idx = [i[1] for i in matched_first]
    unmasked_pred_idx = [i for i in range(cost_matrix.shape[1]) if i not in masked_pred_idx]

    merges_gt_dict = {}
    merges_pred_dict = {}
    merged_gt_subsets = []

    for gt_idx in unmasked_gt_idx:
        check_merge_subset = []
        merged_dist = []

        for pred_idx in unmasked_pred_idx:
            step = 1
            merged_pred = [norm_pred_lines[pred_idx]]

            while True:
                if pred_idx + step in masked_pred_idx or pred_idx + step >= len(norm_pred_lines):
                    break
                else:
                    merged_pred.append(norm_pred_lines[pred_idx + step])
                    merged_pred_flag, continue_flag = judge_pred_merge([norm_gt_lines[gt_idx]], merged_pred)
                    if not merged_pred_flag:
                        break
                    else:
                        step += 1
                    if not continue_flag:
                        break

            check_merge_subset.append(list(range(pred_idx, pred_idx + step)))
            matched_line = ' '.join([norm_pred_lines[i] for i in range(pred_idx, pred_idx + step)])
            dist = Levenshtein_distance(norm_gt_lines[gt_idx], matched_line) / max(len(matched_line), len(norm_gt_lines[gt_idx]))
            merged_dist.append(dist)

        if not merged_dist:
            subset_certain = []
            min_cost_idx = ""
            min_cost = float('inf')
        else:
            min_cost = min(merged_dist)
            min_cost_idx = merged_dist.index(min_cost)
            subset_certain = check_merge_subset[min_cost_idx]

        merges_gt_dict[gt_idx] = {
            'merge_subset': check_merge_subset,
            'merged_cost': merged_dist,
            'min_cost_idx': min_cost_idx,
            'subset_certain': subset_certain,
            'min_cost': min_cost
        }

    subset_certain = [merges_gt_dict[gt_idx]['subset_certain'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]
    subset_certain_cost = [merges_gt_dict[gt_idx]['min_cost'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]

    subset_certain_final = get_final_subset(subset_certain, subset_certain_cost)

    if not subset_certain_final:
        return cost_matrix, norm_pred_lines, range(len(norm_pred_lines))

    final_pred_idx_list = merge_lists_with_sublists(range(len(norm_pred_lines)), subset_certain_final)
    final_norm_pred_lines = [' '.join(norm_pred_lines[idx_list[0]:idx_list[-1]+1]) if isinstance(idx_list, list) else norm_pred_lines[idx_list] for idx_list in final_pred_idx_list]

    new_cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, final_norm_pred_lines)

    return new_cost_matrix, final_norm_pred_lines, final_pred_idx_list

def cal_move_dist(gt, pred):
    assert len(gt) == len(pred), 'Not right length'
    step = 0
    for i, gt_c in enumerate(gt):
        if gt_c != pred[i]:
            step += abs(i - pred.index(gt_c))
            pred[i], pred[pred.index(gt_c)] = pred[pred.index(gt_c)], pred[i]
    return step / len(gt)

def cal_final_match(cost_matrix, norm_gt_lines, norm_pred_lines):
    min_indice = cost_matrix.argmax(axis=1)

    new_cost_matrix, final_norm_pred_lines, final_pred_idx_list = deal_with_truncated(cost_matrix, norm_gt_lines, norm_pred_lines)

    row_ind, col_ind = linear_sum_assignment(new_cost_matrix)

    cost_list = [new_cost_matrix[r][c] for r, c in zip(row_ind, col_ind)]
    matched_col_idx = [final_pred_idx_list[i] for i in col_ind]

    return matched_col_idx, row_ind, cost_list

def initialize_indices(norm_gt_lines, norm_pred_lines):
    gt_lens_dict = {idx: len(gt_line) for idx, gt_line in enumerate(norm_gt_lines)}
    pred_lens_dict = {idx: len(pred_line) for idx, pred_line in enumerate(norm_pred_lines)}
    return gt_lens_dict, pred_lens_dict

def process_matches(matched_col_idx, row_ind, cost_list, norm_gt_lines, norm_pred_lines, pred_lines):
    matches = {}
    unmatched_gt_indices = []
    unmatched_pred_indices = []

    for i in range(len(norm_gt_lines)):
        if i in row_ind:
            idx = list(row_ind).index(i)
            pred_idx = matched_col_idx[idx]

            if pred_idx is None or (isinstance(pred_idx, list) and None in pred_idx):
                unmatched_pred_indices.append(pred_idx)
                continue

            if isinstance(pred_idx, list):
                pred_line = ' | '.join(norm_pred_lines[pred_idx[0]:pred_idx[-1]+1])
                ori_pred_line = ' | '.join(pred_lines[pred_idx[0]:pred_idx[-1]+1])
                matched_pred_indices_range = list(range(pred_idx[0], pred_idx[-1]+1))
            else:
                pred_line = norm_pred_lines[pred_idx]
                ori_pred_line = pred_lines[pred_idx]
                matched_pred_indices_range = [pred_idx]

            edit = cost_list[idx]

            if edit > 0.7:
                unmatched_pred_indices.extend(matched_pred_indices_range)
                unmatched_gt_indices.append(i)
            else:
                matches[i] = {
                    'pred_indices': matched_pred_indices_range,
                    'edit_distance': edit,
                }
                for matched_pred_idx in matched_pred_indices_range:
                    if matched_pred_idx in unmatched_pred_indices:
                        unmatched_pred_indices.remove(matched_pred_idx)
        else:
            unmatched_gt_indices.append(i)

    return matches, unmatched_gt_indices, unmatched_pred_indices

def fuzzy_match_unmatched_items(unmatched_gt_indices, norm_gt_lines, norm_pred_lines):
    matching_dict = {}

    for pred_idx, pred_content in enumerate(norm_pred_lines):
        if isinstance(pred_idx, list):
            continue

        matching_indices = []

        for unmatched_gt_idx in unmatched_gt_indices:
            gt_content = norm_gt_lines[unmatched_gt_idx]
            cur_fuzzy_dist_unmatch, cur_pos, gt_lens, matched_field = sub_gt_fuzzy_matching(pred_content, gt_content)
            if cur_fuzzy_dist_unmatch < 0.4:
                matching_indices.append(unmatched_gt_idx)

        if matching_indices:
            matching_dict[pred_idx] = matching_indices

    return matching_dict

def merge_matches(matches, matching_dict):
    final_matches = {}
    processed_gt_indices = set()

    for gt_idx, match_info in matches.items():
        pred_indices = match_info['pred_indices']
        edit_distance = match_info['edit_distance']

        pred_key = tuple(sorted(pred_indices))

        if pred_key in final_matches:
            if gt_idx not in processed_gt_indices:
                final_matches[pred_key]['gt_indices'].append(gt_idx)
                processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx],
                'edit_distance': edit_distance
            }
            processed_gt_indices.add(gt_idx)

    for pred_idx, gt_indices in matching_dict.items():
        pred_key = (pred_idx,) if not isinstance(pred_idx, (list, tuple)) else tuple(sorted(pred_idx))

        if pred_key in final_matches:
            for gt_idx in gt_indices:
                if gt_idx not in processed_gt_indices:
                    final_matches[pred_key]['gt_indices'].append(gt_idx)
                    processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx for gt_idx in gt_indices if gt_idx not in processed_gt_indices],
                'edit_distance': None
            }
            processed_gt_indices.update(final_matches[pred_key]['gt_indices'])

    return final_matches



def recalculate_edit_distances(final_matches, gt_lens_dict, norm_gt_lines, norm_pred_lines):
    for pred_key, info in final_matches.items():
        gt_indices = sorted(set(info['gt_indices']))

        if not gt_indices:
            info['edit_distance'] = 1
            continue

        if len(gt_indices) > 1:
            merged_gt_content = ''.join(norm_gt_lines[gt_idx] for gt_idx in gt_indices)
            pred_content = norm_pred_lines[pred_key[0]] if isinstance(pred_key[0], int) else ''

            try:
                edit_distance = Levenshtein_distance(merged_gt_content, pred_content)
                normalized_edit_distance = edit_distance / max(len(merged_gt_content), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1

            info['edit_distance'] = normalized_edit_distance
        else:
            gt_idx = gt_indices[0]
            pred_content = ' '.join(norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int))

            try:
                edit_distance = Levenshtein_distance(norm_gt_lines[gt_idx], pred_content)
                normalized_edit_distance = edit_distance / max(len(norm_gt_lines[gt_idx]), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1

            info['edit_distance'] = normalized_edit_distance
            info['pred_content'] = pred_content


def convert_final_matches(final_matches, norm_gt_lines, norm_pred_lines):
    converted_results = []

    all_gt_indices = set(range(len(norm_gt_lines)))
    all_pred_indices = set(range(len(norm_pred_lines)))

    for pred_key, info in final_matches.items():
        pred_content = ' '.join(norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int))

        for gt_idx in sorted(set(info['gt_indices'])):
            result_entry = {
                'gt_idx': int(gt_idx),
                'gt': norm_gt_lines[gt_idx],
                'pred_idx': list(pred_key),
                'pred': pred_content,
                'edit': info['edit_distance']
            }
            converted_results.append(result_entry)

    matched_gt_indices = set().union(*[set(info['gt_indices']) for info in final_matches.values()])
    unmatched_gt_indices = all_gt_indices - matched_gt_indices
    matched_pred_indices = set(idx for pred_key in final_matches.keys() for idx in pred_key if isinstance(idx, int))
    unmatched_pred_indices = all_pred_indices - matched_pred_indices

    if unmatched_pred_indices:
        if unmatched_gt_indices:
            distance_matrix = [
                [Levenshtein_distance(norm_gt_lines[gt_idx], norm_pred_lines[pred_idx]) for pred_idx in unmatched_pred_indices]
                for gt_idx in unmatched_gt_indices
            ]

            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            for i, j in zip(row_ind, col_ind):
                gt_idx = list(unmatched_gt_indices)[i]
                pred_idx = list(unmatched_pred_indices)[j]
                result_entry = {
                    'gt_idx': int(gt_idx),
                    'gt': norm_gt_lines[gt_idx],
                    'pred_idx': [pred_idx],
                    'pred': norm_pred_lines[pred_idx],
                    'edit': 1
                }
                converted_results.append(result_entry)

            matched_gt_indices.update(list(unmatched_gt_indices)[i] for i in row_ind)
        else:
            result_entry = {
                'gt_idx': "",
                'gt': '',
                'pred_idx': list(unmatched_pred_indices),
                'pred': ' '.join(norm_pred_lines[pred_idx] for pred_idx in unmatched_pred_indices),
                'edit': 1
            }
            converted_results.append(result_entry)
    else:
        for gt_idx in unmatched_gt_indices:
            result_entry = {
                'gt_idx': int(gt_idx),
                'gt': norm_gt_lines[gt_idx],
                'pred_idx': "",
                'pred': '',
                'edit': 1
            }
            converted_results.append(result_entry)

    return converted_results

import json


def read_md_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    return content

def save_paired_result(preds, gts, save_path):
    save_result = []
    formula_id = 0
    for gt, pred in zip(gts, preds):
        save_result.append({
            "gt": gt,
            "pred": pred,
            "img_id": formula_id
        })
        formula_id += 1
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)


import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

font = fm.FontProperties(fname=r'font/SimHei.ttf')


def print_aligned_dict(data):
    # Find the maximum length of all keys
    max_key_length = max(len(key) for key in data['testcase1'])

    # Print header
    print(f"{' ' * (max_key_length + 4)}", end="")
    for key in data:
        print(f"{key:>{max_key_length}}", end="")
    print()

    # Print dictionary content
    for subkey in data['testcase1']:
        print(f"{subkey:<{max_key_length + 4}}", end="")
        for key in data:
            print(f"{data[key][subkey]:>{max_key_length}}", end="")
        print()
def create_dict_from_folders(directory):
    body = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            body[folder_name] = {}
    return body


def create_radar_chart(df, title, filename):
    labels = df.columns

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Initialize radar chart
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True), dpi=200)
    # ax.spines['polar'].set_visible(False)

    # Draw radar chart for each dataset
    for index, row in df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.1)
        ax.plot(angles, values, label=index)

        # Add percentage labels next to each data point
        for angle, value in zip(angles, values):
            ax.text(angle, value, '{:.1%}'.format(value), ha='center', va='center', fontsize=7, alpha=0.7)

    # Set labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=font)
    ax.spines['polar'].set_visible(False)  # Hide the outermost circle
    ax.grid(False)
    for j in np.arange(0, 1.2, 0.2):
        ax.plot(angles, len(values) * [j], '-.', lw=0.5, color='black', alpha=0.5)
    for j in range(len(values)):
        ax.plot([angles[j], angles[j]], [0, 1], '-.', lw=0.5, color='black', alpha=0.5)

    # Add title and legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    ax.tick_params(pad=30)
    ax.set_theta_zero_location('N')
    # Save chart to file
    plt.savefig(filename)

# The function is from https://github.com/intsig-textin/markdown_tester
def markdown_to_html(markdown_table):
    rows = [row.strip() for row in markdown_table.strip().split('\n')]
    num_columns = len(rows[0].split('|')) - 2

    html_table = '<table>\n  <thead>\n    <tr>\n'

    header_cells = [cell.strip() for cell in rows[0].split('|')[1:-1]]
    for cell in header_cells:
        html_table += f'      <th>{cell}</th>\n'
    html_table += '    </tr>\n  </thead>\n  <tbody>\n'

    for row in rows[2:]:
        cells = [cell.strip() for cell in row.split('|')[1:-1]]
        html_table += '    <tr>\n'
        for cell in cells:
            html_table += f'      <td>{cell}</td>\n'
        html_table += '    </tr>\n'

    html_table += '  </tbody>\n</table>\n'
    return html_table
def convert_markdown_to_html(self, markdown_content, md_type):
    # Define a regex pattern to find Markdown tables with newlines
    markdown_content = markdown_content.replace('\r', '')
    pattern = re.compile(r'\|\s*.*?\s*\|\n', re.DOTALL)

    # Find all matches in the Markdown content
    matches = pattern.findall(markdown_content)
    for match in matches:
        html_table = markdown_to_html(match)
        markdown_content = markdown_content.replace(match, html_table, 1)  # Only replace the first occurrence
    res_html = convert_table(replace_table_with_placeholder(markdown_content))

    return res_html
def convert_table_str(s):
    s = re.sub(r'<table.*?>','<table>',s)
    s = re.sub(r'<th','<td',s)
    s = re.sub(r'</th>','</td>',s)
    # s = re.sub(r'<td rowspan="(.)">',lambda x:f'<td colspan="1" rowspan="{x.group(1)}">',s)
    # s = re.sub(r'<td colspan="(.)">',lambda x:f'<td colspan="{x.group(1)}" rowspan="1">',s)
    res = ''
    res += '\n\n'
    temp_item = ''
    for c in s:
        temp_item += c
        if c == '>' and not re.search(r'<td.*?>\$',temp_item):
            res += temp_item+'\n'
            temp_item = ''
    return res+'\n'
def merge_table(md):
    table_temp = ''
    for line in md:
        table_temp += line
    return convert_table_str(table_temp)
def find_md_table_mode(line):
    if re.search(r'-*?:',line) or re.search(r'---',line) or re.search(r':-*?',line):
        return True
    return False
def delete_table_and_body(input_list):
    res = []
    for line in input_list:
        if not re.search(r'</?t(able|head|body)>',line):
            res.append(line)
    return res
def merge_tables(input_str):
    # Delete HTML comments
    input_str = re.sub(r'<!--[\s\S]*?-->', '', input_str)

    # Use regex to find each <table> block
    table_blocks = re.findall(r'<table>[\s\S]*?</table>', input_str)

    # Process each <table> block, replace <th> with <td>
    output_lines = []
    for block in table_blocks:
        block_lines = block.split('\n')
        for i, line in enumerate(block_lines):
            if '<th>' in line:
                block_lines[i] = line.replace('<th>', '<td>').replace('</th>', '</td>')
        final_tr = delete_table_and_body(block_lines)
        if len(final_tr) > 2:
            output_lines.extend(final_tr)  # Ignore <table> and </table> tags, keep only table content

    # Rejoin the processed strings
    merged_output = '<table>\n{}\n</table>'.format('\n'.join(output_lines))

    return "\n\n" + merged_output + "\n\n"

def replace_table_with_placeholder(input_string):
    lines = input_string.split('\n')
    output_lines = []

    in_table_block = False
    temp_block = ""
    last_line = ""

    org_table_list = []
    in_org_table = False

    for idx, line in enumerate(lines):
        # if not in_org_table:
        # if "<table>" not in last_line and in_table_block == False and temp_block != "":
        #     output_lines.append(merge_tables(temp_block))
        #     temp_block = ""
        if "<table>" in line:
            # if "<table><tr" in line:
            #     org_table_list.append(line)
            #     in_org_table = True
            #     output_lines.append(last_line)
            #     continue
            # else:
            in_table_block = True
            temp_block += last_line
        elif in_table_block:
            if not find_md_table_mode(last_line) and "</thead>" not in last_line:
                temp_block += "\n" + last_line
            if "</table>" in last_line:
                if "<table>" not in line:
                    in_table_block = False
                    output_lines.append(merge_tables(temp_block))
                    temp_block = ""
        else:
            output_lines.append(last_line)

        last_line = line
        # else:
        #     org_table_list.append(line)
        #     if "</table" in line:
        #         in_org_table = False
        #         last_line = merge_table(org_table_list)
        #         org_table_list = []

    if last_line:
        if in_table_block or "</table>" in last_line:
            temp_block += "\n" + last_line
            output_lines.append(merge_tables(temp_block))
        else:
            output_lines.append(last_line)
    # if "</table>" in last_line:
    #     output_lines.append(merge_tables(temp_block))

    return '\n'.join(output_lines)

def convert_table(input_str):
    # Replace <table>
    output_str = input_str.replace("<table>", "<table border=\"1\" >")

    # Replace <td>
    output_str = output_str.replace("<td>", "<td colspan=\"1\" rowspan=\"1\">")

    return output_str

def convert_markdown_to_html(markdown_content):
    # Define a regex pattern to find Markdown tables with newlines
    markdown_content = markdown_content.replace('\r', '')+'\n'
    pattern = re.compile(r'\|\s*.*?\s*\|\n', re.DOTALL)

    # Find all matches in the Markdown content
    matches = pattern.findall(markdown_content)

    for match in matches:
        html_table = markdown_to_html(match)
        markdown_content = markdown_content.replace(match, html_table, 1)  # Only replace the first occurrence

    res_html = convert_table(replace_table_with_placeholder(markdown_content))

    return res_html
