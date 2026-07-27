# flake8: noqa
# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

import ast
import json
import re
import string
import sys
from collections import deque
from itertools import product
from typing import Any, Callable, Optional, Sequence

import distance
import editdistance
import ipdb
import Levenshtein
import numpy as np
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from tqdm import tqdm
from zss import Node, simple_distance

from vlmeval.dataset.utils.Ocrbench_v2.parallel import parallel_process

sys.set_int_max_str_digits(1000000)
class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            #inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]} for filename in samples]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores


def convert_table_to_html_str(table_row_list=[]):
    """
    Given a list of table rows, build the corresponding html string, which is used to compute the TEDS score.
    We use the official code of PubTabNet to compute TEDS score, it does not consider '<th>' label.
    We also remove unneccessary spaces within a table cell and extra '\n' as they will influence the TEDS score.
    """
    html_table_str = "<html><body><table>" + '\n'
    for data_row in table_row_list:
        html_table_str += "<tr>"
        for cell_str in data_row:
            html_table_str += f"<td>{cell_str}</td>"
        html_table_str += "</tr>"
        html_table_str += '\n'
    html_table_str += "</table></body></html>"
    html_table_str = html_table_str.replace('\n','')
    return html_table_str


def convert_markdown_table_to_html(markdown_table):
    """
    Converts a markdown table to the corresponding html string for TEDS computation.
    """
    # remove extra code block tokens like '```markdown' and '```
    markdown_table = markdown_table.strip('```markdown').strip('```').strip()
    row_str_list = markdown_table.split('\n')
    # extra the first header row and other data rows
    valid_row_str_list = [row_str_list[0]]+row_str_list[2:]
    table_rows = []
    for row_str in valid_row_str_list:
        one_row = []
        for cell in row_str.strip().split('|')[1:-1]:
            if set(cell) != set(' '):
                one_row.append(cell.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    # build html string based on table rows
    html_str = convert_table_to_html_str(table_rows)
    return html_str


def dict_to_html(data):
    html = "<html><body><table>\n"
    for key, value in data.items():
        if not isinstance(value, str):
            value = str(value)
        value_str = ' '.join(value)

        html += f"  <tr><td>{key}</td><td>{value_str}</td></tr>\n"
    html += "</table></body></html>"
    return html


def convert_str_to_dict(predict_str: str):
    """
    Parses the 'predict' string and returns a dictionary.
    Missing or unparseable content is handled gracefully.

    Parameters:
    - predict_str (str): The prediction string containing the output dict.

    Returns:
    - dict: A dictionary extracted from the predict string.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    match = re.search(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        content = predict_str.strip()

    data = {}
    success = False

    # try parsing with JSON
    try:
        data = json.loads(content)
        success = True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # try parsing with ast.literal_eval
    if not success:
        try:
            data = ast.literal_eval(content)
            if isinstance(data, dict):
                success = True
        except (ValueError, SyntaxError, TypeError):
            pass

    # try parsing with regex
    if not success:
        key_value_pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
        matches = re.findall(key_value_pattern, content)
        try:
            for key, value in matches:
                data[key.strip()] = value.strip()
        except:
            return {}

    if not data:
        return {}

    try:
        result = {k.strip(): str(v).strip() for k, v in data.items()}
    except:
        return {}
    return result


def convert_str_to_multi_dict(predict_str: str):
    """
    Parses the 'predict' string and returns a dictionary.
    Handles nested dictionaries and missing or unparseable content gracefully.

    Parameters:
    - predict_str (str): The prediction string containing the output dict.

    Returns:
    - dict: A dictionary extracted from the predict string.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    matches = re.findall(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if matches:
        content = max(matches, key=len)
    else:
        content = predict_str.strip()

    def strip_variable_assignment(s):
        variable_assignment_pattern = r'^\s*\w+\s*=\s*'
        return re.sub(variable_assignment_pattern, '', s.strip(), count=1)

    content = strip_variable_assignment(content)

    def remove_comments(s):
        return re.sub(r'#.*', '', s)

    content = remove_comments(content)

    last_brace_pos = content.rfind('}')
    if last_brace_pos != -1:
        content = content[:last_brace_pos+1]

    data = {}
    success = False

    # try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(content)
        if isinstance(data, dict):
            success = True
    except (ValueError, SyntaxError, TypeError):
        pass

    if not success:
        return {}

    def process_data(obj):
        if isinstance(obj, dict):
            return {k: process_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_data(elem) for elem in obj]
        else:
            return obj

    data = process_data(data)

    return data


def generate_combinations(input_dict):
    """
    Function to generate all possible combinations of values from a dictionary.
    """
    kie_answer = input_dict
    if not isinstance(kie_answer, dict):
        kie_answer = kie_answer.strip('"')
        try:
            kie_answer = json.loads(kie_answer)
        except json.JSONDecodeError:
            try:
                kie_answer = ast.literal_eval(kie_answer)
                if not isinstance(kie_answer, dict):
                    kie_answer = ast.literal_eval(kie_answer)
            except (ValueError, SyntaxError, TypeError):
                print(f"Unable to parse 'answers' field: {kie_answer}")
                return {}

        # Ensure the parsed result is a dictionary.
        if not isinstance(kie_answer, dict):
            print("Parsed 'answers' is still not a dictionary.")
            raise ValueError("Input could not be parsed into a dictionary.")

        keys = list(kie_answer.keys())

        value_lists = []
        for single_key in keys:
            sinlge_value = kie_answer[single_key]
            if not isinstance(sinlge_value, list):
                sinlge_value = [sinlge_value]
            value_lists.append(sinlge_value)

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result

    else:
        keys = list(input_dict.keys())
        value_lists = [input_dict[key] for key in keys]

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result


def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score for KIE task between predicted and ground truth dictionaries.

    Args:
        preds (dict): The predicted key-value pairs.
        gts (dict): The ground truth key-value pairs.
        ignores (list): The list of keys to ignore during evaluation.

    Returns:
        dict: A dictionary where keys are field names and values are their corresponding F1-scores.
    """
    # Optionally remove ignored keys from predictions and ground truths
    keys = set(preds.keys()).union(set(gts.keys())) - set(ignores)
    f1_scores = {}

    for key in keys:
        pred_value = preds.get(key, None)
        gt_value = gts.get(key, None)

        if pred_value:
            pred_value = pred_value.lower().strip().replace("\n"," ").replace(" ", "")
        if gt_value:
            gt_value = gt_value.lower().strip().replace("\n"," ").replace(" ", "")

        if pred_value is None and gt_value is None:
            continue
        elif pred_value is None:
            precision = 0.0
            recall = 0.0
        elif gt_value is None:
            # false positive
            precision = 0.0
            recall = 0.0
        else:
            if pred_value == gt_value:
                # True positive
                precision = 1.0
                recall = 1.0
            else:
                precision = 0.0
                recall = 0.0

        # Compute F1-score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[key] = f1_score

    if len(f1_scores) == 0:
        return 0
    average_f1 = sum(f1_scores.values()) / len(f1_scores)

    return average_f1


def pre_clean(text):
    text = re.sub(r'<bos>|<eos>|<pad>|<unk>', '', text)
    text = re.sub(r'\s##(\S)', r'\1', text)
    text = re.sub(r'\\\s', r'\\', text)
    text = re.sub(r'\s\*\s\*\s', r'**', text)
    text = re.sub(r'{\s', r'{', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\\begin\s', r'\\begin', text)
    text = re.sub(r'\\end\s', r'\\end', text)
    text = re.sub(r'\\end{table}', r'\\end{table} \n\n', text)
    text = text.replace('\n', ' ')
    text = text.replace('*', ' ')
    text = text.replace('_', ' ')
    return text


def get_tree(input_str):
    tree = (Node('ROOT').addkid(Node('TITLE')))

    lines = input_str.split("\n")
    lines = [pre_clean(line) for line in lines]
    last_title = ''
    for line in lines:
        if line.startswith('#'):
            child = tree.get('ROOT')
            line = line.replace('#', '')
            child.addkid(Node(line))
            last_title = line
        else:
            if last_title == '':
                child = tree.get('TITLE')
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)
                child.addkid(Node(line))
    return tree

def STEDS(pred_tree, ref_tree):
    def my_distance(pred, ref):
        if len(pred.split()) == 0 or len(ref.split()) == 0:
            return 1
        else:
            return 0
    total_distance = simple_distance(pred_tree, ref_tree, label_dist=my_distance)
    num_of_nodes = max(len(list(pred_tree.iter())), len(list(ref_tree.iter())))
    return 1-total_distance/num_of_nodes


def doc_parsing_evaluation(pred, gt):
    score = 0
    if not isinstance(pred, str):
        return 0
    pred_tree = get_tree(pred)
    gt_tree = get_tree(gt)
    score = STEDS(pred_tree, gt_tree)

    return score


def wrap_html_table(html_table):
    """
    The TEDS computation from PubTabNet code requires that the input html table should have <html>, <body>, and <table> tags.
    Add them if they are missing.
    """
    html_table = html_table.replace('\n','')
    # add missing <table> tag if missing
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"
    else:
        pass
    # add <body> and <html> tags if missing
    if '<body>' not in html_table:
        html_table = '<body>' + html_table + '</body>'
    if '<html>' not in html_table:
        html_table = '<html>' + html_table + '</html>'
    return html_table


def get_anls(s1, s2):
    try:
        s1 = s1.lower()
        s2 = s2.lower()
    except:
        pass
    if s1 == s2:
        return 1.0
    iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
    anls = iou
    return anls


def ocr_eval(references,predictions):
    socre_=0.0
    None_num=0
    for idx,ref_value in enumerate(references):
        pred_value = predictions[idx]
        pred_values, ref_values = [], []
        if isinstance(pred_value, str):
            pred_values.append(pred_value)
        else:
            pred_values = pred_value
        if isinstance(ref_value, str):
            ref_values.append(ref_value)
        else:
            ref_values = ref_value

        temp_score = 0.0
        temp_num = len(ref_values)

        for tmpidx, tmpref in enumerate(ref_values):
            tmppred = pred_values[tmpidx] if tmpidx < len(pred_values) else pred_values[0]
            if len(pred_values) == 1 and tmppred != "None" and "None" not in ref_values:  # pred 1, and not None
                temp_score = max(temp_score, get_anls(tmppred, tmpref))
                temp_num = len(ref_values)
            else:
                if tmppred=='None' and tmpref!='None':
                    temp_score += 0.0
                elif tmpref=='None':
                    temp_num -= 1
                else:
                    temp_score += get_anls(tmppred, tmpref)
        if temp_num == 0:
            ocr_score = 0.0
            None_num += 1
        else:
            ocr_score = temp_score / (temp_num)
        socre_ += ocr_score
    if None_num == len(references):
        return 9999
    else:
        return round(socre_ / (len(references)-None_num), 5)


def csv_eval(predictions,references,easy, pred_type='json'):
    predictions = predictions
    labels = references
    def is_int(val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    def is_float(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    def convert_dict_to_list(data):
        """
        Convert a dictionary to a list of tuples, handling both simple and nested dictionaries.

        Args:
        data (dict): The input dictionary, which might be nested or simple.

        Returns:
        list: A list of tuples generated from the input dictionary.
        """
        # print(data)
        converted_list = []
        for key, value in data.items():
            # Check if the value is a dictionary (indicating a nested structure)
            if isinstance(value, dict):
                # Handle nested dictionary
                for subkey, subvalue in value.items():
                    # converted_list.append((key, subkey, subvalue))
                    converted_list.append((key, subkey, re.sub(r'[^\d.-]', '', str(subvalue))))

            else:
                # Handle simple key-value pair
                # converted_list.append((key, "value", value))
                converted_list.append((key, "value", re.sub(r'[^\d.-]', '', str(value))))
        return converted_list


    def csv2triples(csv, separator='\\t', delimiter='\\n'):
        lines = csv.strip().split(delimiter)
        header = lines[0].split(separator)
        triples = []
        for line in lines[1:]:
            if not line:
                continue
            values = line.split(separator)
            entity = values[0]
            for i in range(1, len(values)):
                if i >= len(header):
                    break
                #---------------------------------------------------------
                temp = [entity.strip(), header[i].strip()]
                temp = [x if len(x)==0 or x[-1] != ':' else x[:-1] for x in temp]
                value = values[i].strip()
                value = re.sub(r'[^\d.-]', '', str(value))
                # value = value.replace("%","")
                # value = value.replace("$","")
                triples.append((temp[0], temp[1], value))
                #---------------------------------------------------------
        return triples

    def csv2triples_noheader(csv, separator='\\t', delimiter='\\n'):
        lines = csv.strip().split(delimiter)
        maybe_header = [x.strip() for x in lines[0].split(separator)]
        not_header = False
        if len(maybe_header) > 2:
            for c in maybe_header[1:]:
                try:
                    num = float(c)
                    not_header = True
                except:
                    continue
                if not_header:
                    break
        header = None if not_header else maybe_header
        data_start = 0 if not_header and separator in lines[0] else 1
        triples = []
        for line in lines[data_start:]:
            if not line:
                continue
            values = [x.strip() for x in line.split(separator)]
            entity = values[0]
            for i in range(1, len(values)):
                try:
                    temp = [entity if entity[-1]!=':' else entity[:-1], ""]
                except:
                    temp = [entity, ""]
                if header is not None:
                    try:
                        this_header = header[i]
                        temp = [entity, this_header]
                        temp = [x if x[-1] != ':' else x[:-1] for x in temp]
                    except:
                        this_header = entity.strip()
                value = values[i].strip()
                value = re.sub(r'[^\d.-]', '', str(value))
                # value = value.replace("%","")
                # value = value.replace("$","")
                triples.append((temp[0], temp[1], value))
                #---------------------------------------------------------
        return triples

    def process_triplets(triplets):
        new_triplets = []
        for triplet in triplets:
            new_triplet = []
            triplet_temp = []
            if len(triplet) > 2:
                if is_int(triplet[2]) or is_float(triplet[2]):
                    triplet_temp = (triplet[0].lower(), triplet[1].lower(), float(triplet[2]))
                else:
                    triplet_temp = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
            else:
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), "no meaning")
            new_triplets.append(triplet_temp)
        return new_triplets

    def intersection_with_tolerance(a, b, tol_word, tol_num):
        a = set(a)
        b = set(b)
        c = set()
        for elem1 in a:
            for elem2 in b:
                if is_float(elem1[-1]) and is_float(elem2[-1]):
                    if ((Levenshtein.distance(''.join(elem1[:-1]),''.join(elem2[:-1])) <= tol_word) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num))or \
                    ((''.join(elem1[:-1]) in ''.join(elem2[:-1])) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num)) or \
                    ((''.join(elem2[:-1]) in ''.join(elem1[:-1])) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num)):
                        c.add(elem1)
                else:
                    if (Levenshtein.distance(''.join([str(i) for i in elem1]),''.join([str(j) for j in elem2])) <= tol_word):
                        c.add(elem1)
        return list(c)

    def union_with_tolerance(a, b, tol_word, tol_num):
        c = set(a) | set(b)
        d = set(a) & set(b)
        e = intersection_with_tolerance(a, b, tol_word, tol_num)
        f = set(e)
        g = c-(f-d)
        return list(g)

    def get_eval_list(pred_csv, label_csv, separator='\\t', delimiter='\\n', tol_word=3, tol_num=0.05, pred_type='json'):

        if pred_type == 'json':
            pred_triple_list=[]
            for it in pred_csv:
                pred_triple_temp = convert_dict_to_list(it)
                pred_triple_pre = process_triplets(pred_triple_temp)
                pred_triple_list.append(pred_triple_pre)
        else:
            pred_triple_list=[]
            for it in pred_csv:
                pred_triple_temp = csv2triples(it, separator=separator, delimiter=delimiter)
                # pred_triple_temp = csv2triples_noheader(it, separator=separator, delimiter=delimiter)
                pred_triple_pre = process_triplets(pred_triple_temp)
                pred_triple_list.append(pred_triple_pre)

        label_triple_list=[]
        for it in label_csv:
            label_triple_temp = convert_dict_to_list(it)
            label_triple_pre = process_triplets(label_triple_temp)
            label_triple_list.append(label_triple_pre)


        intersection_list=[]
        union_list=[]
        sim_list=[]
        # for each chart image
        for pred,label in zip(pred_triple_list, label_triple_list):
            for idx in range(len(pred)):
                try:
                    if label[idx][1] == "value" and "value" not in pred[idx][:2]:
                        pred[idx] = (pred[idx][0], "value", pred[idx][2])
                    temp_pred_head = sorted(pred[idx][:2])
                    temp_gt_head = sorted(label[idx][:2])
                    pred[idx] = (temp_pred_head[0], temp_pred_head[1], pred[idx][2])
                    label[idx] = (temp_gt_head[0], temp_gt_head[1], label[idx][2])
                except:
                    continue
            intersection = intersection_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
            union = union_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
            sim = len(intersection)/len(union)
            intersection_list.append(intersection)
            union_list.append(union)
            sim_list.append(sim)
        return intersection_list, union_list, sim_list

    def get_ap(predictions, labels, sim_threhold, tolerance, separator='\\t', delimiter='\\n', easy=1):
        if tolerance == 'strict':
            tol_word=0
            if easy == 1:
                tol_num=0
            else:
                tol_num=0.1

        elif tolerance == 'slight':
            tol_word=2
            if easy == 1:
                tol_num=0.05
            else:
                tol_num=0.3

        elif tolerance == 'high':
            tol_word= 5
            if easy == 1:
                tol_num=0.1
            else:
                tol_num=0.5
        intersection_list, union_list, sim_list = get_eval_list(predictions, labels, separator=separator, delimiter=delimiter, tol_word=tol_word, tol_num=tol_num, pred_type=pred_type)
        ap = len([num for num in sim_list if num >= sim_threhold])/(len(sim_list)+1e-16)
        return ap

    map_strict = 0
    map_slight = 0
    map_high = 0
    s="\\t"
    d="\\n"

    for sim_threhold in np.arange (0.5, 1, 0.05):
        map_temp_strict = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='strict', separator=s, delimiter=d, easy=easy)
        map_temp_slight = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='slight', separator=s, delimiter=d, easy=easy)
        map_temp_high = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='high', separator=s, delimiter=d, easy=easy)
        map_strict += map_temp_strict/10
        map_slight += map_temp_slight/10
        map_high += map_temp_high/10

    em = get_ap(predictions, labels, sim_threhold=1, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_50_strict = get_ap(predictions, labels, sim_threhold=0.5, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_75_strict = get_ap(predictions, labels, sim_threhold=0.75, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_90_strict = get_ap(predictions, labels, sim_threhold=0.90, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_50_slight = get_ap(predictions, labels, sim_threhold=0.5, tolerance='slight', separator=s, delimiter=d, easy=easy)
    ap_75_slight = get_ap(predictions, labels, sim_threhold=0.75, tolerance='slight', separator=s, delimiter=d, easy=easy)
    ap_90_slight = get_ap(predictions, labels, sim_threhold=0.90, tolerance='slight', separator=s, delimiter=d, easy=easy)
    ap_50_high = get_ap(predictions, labels, sim_threhold=0.5, tolerance='high', separator=s, delimiter=d, easy=easy)
    ap_75_high = get_ap(predictions, labels, sim_threhold=0.75, tolerance='high', separator=s, delimiter=d, easy=easy)
    ap_90_high = get_ap(predictions, labels, sim_threhold=0.90, tolerance='high', separator=s, delimiter=d, easy=easy)


    return em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high

def draw_SCRM_table(em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high,title_ocr_socre,source_ocr_socre,x_title_ocr_socre,y_title_ocr_socre,structure_accuracy):

    result=f'''
            -----------------------------------------------------------\n
            |  Metrics   |  Sim_threshold  |  Tolerance  |    Value    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % map_strict}    |     \n
            |             |                 ----------------------------\n
            |  mPrecison  |  0.5:0.05:0.95  |   slight    |    {'%.4f' % map_slight}    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % map_high}    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % ap_50_strict}    |\n
            |             |                  ---------------------------\n
            |  Precison   |       0.5       |   slight    |    {'%.4f' % ap_50_slight }    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % ap_50_high }    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % ap_75_strict}    |\n
            |             |                  ---------------------------\n
            |  Precison   |      0.75       |   slight    |    {'%.4f' % ap_75_slight}    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % ap_75_high}    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % ap_90_strict}    |\n
            |             |                  ---------------------------\n
            |  Precison   |       0.9       |   slight    |    {'%.4f' % ap_90_slight }    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % ap_90_high}    |\n
            -----------------------------------------------------------\n
            |Precison(EM) |                                    {'%.4f' % em}    |\n
            -----------------------------------------------------------\n
            |Title(EM)    |                                    {'%.4f' % title_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |Source(EM)   |                                    {'%.4f' % source_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |X_title(EM)  |                                    {'%.4f' % x_title_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |Y_title(EM)  |                                    {'%.4f' % y_title_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |structure_acc|                                    {'%.4f' % structure_accuracy}    |\n
            -----------------------------------------------------------\n


            '''
    return result


if __name__ == '__main__':
    import json
    import pprint

    # markdown structure for Table Parsing task
    pred_markdown = "| 1 | august 5 , 1972 | detroit lions | l 23 - 31 | 0 - 1 |\n| 2 | august 12 , 1972 | green bay packers | l 13 - 14 | 0 - 2 |\n| 3 | august 19 , 1972 | cincinnati bengals | w 35 - 17 | 1 - 2 |\n| 4 | august 25 , 1972 | atlanta falcons | w 24 - 10 | 2 - 2 |\n| 5 | august 31 , 1972 | washington redskins | l 24 - 27 | 2 - 3 |\n| 6 | september 10 , 1972 | minnesota vikings | w 21 - 19 | 3 - 3 |"
    true_markdown = "| week | date | opponent | result | record |\n| --- | --- | --- | --- | --- |\n| 1 | august 5 , 1972 | detroit lions | l 23 - 31 | 0 - 1 |\n| 2 | august 12 , 1972 | green bay packers | l 13 - 14 | 0 - 2 |\n| 3 | august 19 , 1972 | cincinnati bengals | w 35 - 17 | 1 - 2 |\n| 4 | august 25 , 1972 | atlanta falcons | w 24 - 10 | 2 - 2 |\n| 5 | august 31 , 1972 | washington redskins | l 24 - 27 | 2 - 3 |\n| 6 | september 10 , 1972 | minnesota vikings | w 21 - 19 | 3 - 3 |"
    teds = TEDS(n_jobs=4)
    pred_table_html = convert_markdown_table_to_html(pred_markdown)
    true_table_html = convert_markdown_table_to_html(true_markdown)

    scores = teds.evaluate(pred_table_html, true_table_html)

    pp = pprint.PrettyPrinter()
    pp.pprint(scores)

    # dict structure for Key Information Extraction task
    pred_dict = {
            "company": [
                "OLD TOWN "
            ],
            "date": [
                "2024"
            ],
            "address": [
                "SRI RAMPAI"
            ],
            "total": [
                "30"
            ]
        }
    true_dict = {
            "company": [
                "OLD TOWN KOPITAM SND BHD"
            ],
            "date": [
                "2024/9/27"
            ],
            "address": [
                "SRI RAMPAI"
            ],
            "total": [
                "30"
            ]
        }
    teds = TEDS(n_jobs=4)
    pred_dict_html = dict_to_html(pred_dict)
    true_dict_html = dict_to_html(true_dict)
    print(pred_dict_html)
    print(true_dict_html)

    scores = teds.evaluate(pred_dict_html, true_dict_html)

    pp = pprint.PrettyPrinter()
    pp.pprint(scores)
