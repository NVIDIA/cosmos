#!/usr/bin/env python3

import json
import re
from typing import Any, Dict, List, Union

import nltk
import nltk.data
from nltk.corpus import words


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(
        self, predicted_answer: Any, ground_truth: Any,
        params: Dict[str, Any]
    ) -> bool:
        raise NotImplementedError


class WordLadderEvaluator(BaseEvaluator):
    """
    评估Word Ladder谜题解答的评估器

    Word Ladder是一种单词转换谜题，要求：
    1. 从起始单词变为目标单词
    2. 每一步只能改变一个字母
    3. 每一步都必须形成有效的英文单词
    4. 找出一条有效路径，步数在指定范围内
    """

    def __init__(self):
        """初始化评估器，下载并加载英语词典"""
        self.english_words = set()
        try:
            # 检查 'words' 语料库是否已经下载
            try:
                nltk.data.find('corpora/words')
            except nltk.downloader.DownloadError:
                nltk.download('words', quiet=True)

            self.english_words = set(words.words())
        except Exception:
            pass

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """准备发送给模型的提示词"""
        # 从initial_state获取谜题信息
        if "initial_state" not in params:
            return question

        initial_state = params["initial_state"]
        start_word = initial_state.get("start_word", "")
        target_word = initial_state.get("target_word", "")
        solution = initial_state.get("solution", [])

        # 如果没有找到必要信息，返回原问题
        if not start_word or not target_word:
            return question

        # 根据solution估计最小和最大步数
        min_steps = max_steps = 0
        if solution and len(solution) > 1:
            path_length = len(solution) - 1
            min_steps = max(1, path_length - 1)
            max_steps = path_length + 1
        else:
            # 默认值
            min_steps = 3
            max_steps = 6

        # 构建提示词
        prompt = (
            f"This is a Word Ladder puzzle. Transform the start word "
            f"into the target word by changing one letter at a time, "
            f"ensuring that each step forms a valid English word. "
            f"Follow these rules:\n\n"
            f"1. Change exactly one letter at a time.\n"
            f"2. Each step must form a valid English word.\n"
            f"3. Find a solution path with {min_steps} to "
            f"{max_steps} steps.\n\n"
            f"Starting word: {start_word}\n"
            f"Target word: {target_word}\n\n"
            f"Please provide the complete solution path as a list of "
            f"words, including the starting and target words.\n"
            f"Example answer format: \"cat -> cot -> cog -> dog\" or "
            f"[\"cat\", \"cot\", \"cog\", \"dog\"]"
        )

        return prompt

    def extract_answer(self, model_output: str) -> List[str]:
        """从模型输出中提取Word Ladder解答路径"""
        if isinstance(model_output, dict) and "text" in model_output:
            model_output = model_output["text"]

        # 尝试多种格式匹配

        # 1. 尝试匹配 JSON 数组格式: ["word1", "word2", ...]
        json_pattern = r'\[\s*"[a-zA-Z]+"(?:\s*,\s*"[a-zA-Z]+")*\s*\]'
        matches = re.findall(json_pattern, model_output)
        if matches:
            try:
                # 解析最后一个匹配的 JSON 数组
                path = json.loads(matches[-1])
                return [word.lower() for word in path]  # 转换为小写
            except Exception:
                pass

        # 2. 尝试匹配单引号数组格式: ['word1', 'word2', ...]
        single_quote_pattern = (
            r'\[\s*\'[a-zA-Z]+\'(?:\s*,\s*\'[a-zA-Z]+\')*\s*\]'
        )
        matches = re.findall(single_quote_pattern, model_output)
        if matches:
            try:
                # 将单引号替换为双引号以便 JSON 解析
                json_str = matches[-1].replace("'", "\"")
                path = json.loads(json_str)
                return [word.lower() for word in path]
            except Exception:
                pass

        # 3. 尝试匹配箭头分隔格式: word1 -> word2 -> word3
        arrow_pattern = r'[a-zA-Z]+(?:\s*->\s*[a-zA-Z]+)+'
        matches = re.findall(arrow_pattern, model_output)
        if matches:
            # 取最长的匹配（通常是最完整的路径）
            best_match = max(matches, key=len)
            # 分割并清理
            path = [word.strip().lower() for word in best_match.split('->')]
            return path

        # 4. 尝试匹配逗号分隔格式: word1, word2, word3
        comma_pattern = r'[a-zA-Z]+(?:\s*,\s*[a-zA-Z]+)+'
        matches = re.findall(comma_pattern, model_output)
        if matches:
            # 取最长的匹配
            best_match = max(matches, key=len)
            # 分割并清理
            path = [word.strip().lower() for word in best_match.split(',')]
            return path

        # 5. 尝试匹配行分隔格式: 每行一个单词
        lines = model_output.split('\n')
        word_lines = []
        for line in lines:
            # 寻找可能是单词的行
            word_match = re.search(r'^[a-zA-Z]+$', line.strip())
            if word_match:
                word_lines.append(line.strip().lower())

        if word_lines and len(word_lines) >= 2:
            return word_lines

        # 6. 尝试提取格式化列表中的单词
        list_item_pattern = (
            r'(?:^|\n)(?:\d+\.\s+|\*\s+|-\s+|\(\d+\)\s+)([a-zA-Z]+)'
        )
        matches = re.findall(list_item_pattern, model_output)
        if matches and len(matches) >= 2:
            return [word.lower() for word in matches]

        # 7. 最后尝试直接提取所有单词
        word_pattern = r'\b[a-zA-Z]{3,}\b'  # 至少3个字母的单词
        all_words = re.findall(word_pattern, model_output)

        # 过滤掉常见的非路径单词
        filtered_words = []
        common_words = {
            "the", "and", "word", "ladder", "solution", "puzzle",
            "step", "steps", "path", "example", "format", "answer"
        }
        for word in all_words:
            if word.lower() not in common_words and len(word) >= 3:
                filtered_words.append(word.lower())

        if filtered_words and len(filtered_words) >= 2:
            return filtered_words

        # 如果所有尝试都失败，返回空列表
        return []

    def _is_valid_word(self, word: str) -> bool:
        """检查单词是否在英语词典中"""
        return word.lower() in self.english_words

    def evaluate(
        self, output: Union[str, List[str]], ground_truth: Any,
        params: Dict[str, Any]
    ) -> bool:
        """
        评估预测的Word Ladder解答是否正确，
        基于规则而不是直接比对ground_truth

        参数:
            output: 模型预测的解答路径（字符串或列表）
            ground_truth: 正确答案（仅作参考，不直接比对）
            params: 包含谜题信息的参数

        返回:
            是否正确（布尔值）
        """
        # 检查output类型，如果是字符串，则提取答案
        if isinstance(output, str):
            pred_path = self.extract_answer(output)
        else:
            # 如果已经是列表形式，直接使用
            pred_path = output

        # 如果预测答案为空或只有一个单词，直接返回False
        if not pred_path or len(pred_path) < 2:
            return False

        # 获取谜题参数
        start_word = params.get("start_word", "").lower()
        target_word = params.get("target_word", "").lower()

        # 如果params中没有直接的start_word和target_word，
        # 尝试从initial_state中获取
        if not start_word or not target_word:
            initial_state = params.get("initial_state", {})
            start_word = initial_state.get("start_word", "").lower()
            target_word = initial_state.get("target_word", "").lower()

        # 如果仍然无法获取起始和目标单词，返回False
        if not start_word or not target_word:
            return False

        # 标准化预测答案，转换为小写
        pred_path = [word.lower() for word in pred_path]

        # 1. 验证路径起点和终点
        if pred_path[0] != start_word:
            return False

        if pred_path[-1] != target_word:
            return False

        # 2. 验证每一步只改变一个字母
        for i in range(len(pred_path) - 1):
            if not self._is_one_letter_apart(
                pred_path[i], pred_path[i + 1]
            ):
                return False

        # 3. 检查路径中是否有重复单词
        if len(pred_path) != len(set(pred_path)):
            return False

        for word in pred_path:
            if not self._is_valid_word(word):
                return False

        # 所有检查都通过，解答正确
        return True

    def _is_one_letter_apart(self, word1: str, word2: str) -> bool:
        """检查两个单词是否只相差一个字母"""
        # 如果长度不同，返回False
        if len(word1) != len(word2):
            return False

        # 计算不同字母的数量
        diff_count = sum(1 for c1, c2 in zip(word1, word2) if c1 != c2)

        # 只有恰好一个字母不同时返回True
        return diff_count == 1
