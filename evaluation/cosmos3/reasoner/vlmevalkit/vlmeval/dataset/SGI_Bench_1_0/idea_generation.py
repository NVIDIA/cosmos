import ast
import json
import os.path as osp
import re
import time
from datetime import datetime
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from datasets import load_dataset

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.smp.log import get_logger
from vlmeval.utils.mp_util import track_progress_rich
from ..text_base import TextBaseDataset
from ..utils.judge_util import build_judge
from .utils import (flip_evaluation_result, format_idea_data, get_context_from_data,
                    get_evaluation_prompt_modified, parse_evaluation_result)

embedding_model = None
logger = get_logger(__name__)


def parse_generated_idea(text: str) -> Dict[str, Any]:
    """Parse the generated research proposal text into a structured dictionary"""
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    json_block_match = re.search(json_block_pattern, text)
    if json_block_match:
        json_str = json_block_match.group(1).strip()
        try:
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError:
            pass
    try:
        parsed_data = json.loads(text)
        return parsed_data
    except json.JSONDecodeError:
        pass
    result = {}
    idea_patterns = [
        r"[\"']?Idea[\"']?\s*:\s*[\"'](.*?)[\"']",
        r"1\.\s*Idea[:\s-]+(.*?)(?=\n\s*(?:2\.|Implementation))",
    ]
    for pattern in idea_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["Idea"] = match.group(1).strip()
            break
    steps_patterns = [
        r"[\"']?ImplementationSteps[\"']?\s*:\s*\{(.*?)\}",
        r"2\.\s*Implementation Steps[:\s-]+(.*?)(?=\n\s*(?:3\.|Implementation Order))",
    ]
    for pattern in steps_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            steps_text = match.group(1).strip()
            steps_dict = {}
            step_matches = re.findall(r"[\"'](\d+)[\"']\s*:\s*[\"'](.*?)[\"']", steps_text)
            for step_num, step_desc in step_matches:
                steps_dict[step_num] = step_desc.strip()
            if steps_dict:
                result["ImplementationSteps"] = steps_dict
                break
    order_patterns = [
        r"[\"']?ImplementationOrder[\"']?\s*:\s*\[(.*?)\]",
        r"3\.\s*Implementation Order[:\s-]+(.*?)(?=\n\s*(?:4\.|Dataset))",
    ]
    for pattern in order_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            order_text = match.group(1).strip()
            order_list = re.findall(r'["\']([^"\']+)["\']', order_text)
            if order_list:
                result["ImplementationOrder"] = order_list
                break
    dataset_patterns = [
        r"[\"']?Dataset[\"']?\s*:\s*[\"'](.*?)[\"'](?=\s*,\s*[\"'])",
        r"4\.\s*Dataset[:\s-]+(.*?)(?=\n\s*(?:5\.|Evaluation))",
    ]
    for pattern in dataset_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["Dataset"] = match.group(1).strip()
            break
    metrics_patterns = [
        r"[\"']?EvaluationMetrics[\"']?\s*:\s*\{(.*?)\}(?=\s*,\s*[\"'])",
        r"5\.\s*Evaluation Metrics[:\s-]+(.*?)(?=\n\s*(?:6\.|Expected))",
    ]
    for pattern in metrics_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            metrics_text = match.group(1).strip()
            metrics_dict = {}
            metric_matches = re.findall(r"[\"']([^\"']+)[\"']\s*:\s*[\"'](.*?)[\"']", metrics_text)
            for metric_name, metric_desc in metric_matches:
                metrics_dict[metric_name.strip()] = metric_desc.strip()
            if metrics_dict:
                result["EvaluationMetrics"] = metrics_dict
                break
    outcome_patterns = [
        r"[\"']?ExpectedOutcome[\"']?\s*:\s*[\"'](.*?)[\"']",
        r"6\.\s*Expected Outcome[:\s-]+(.*?)$",
    ]
    for pattern in outcome_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["ExpectedOutcome"] = match.group(1).strip()
            break
    if not result:
        result["full_text"] = text
    return result


# 所有原始工具函数
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1, a_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    return np.dot(a_normalized, b_normalized.T)


def edge_jaccard(G1, G2):
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    if not edges1 and not edges2:
        return 1.0
    return len(edges1 & edges2) / len(edges1 | edges2)


def node_text_similarity(G1, G2):
    texts1 = [G1.nodes[n]['text'] for n in G1.nodes()]
    texts2 = [G2.nodes[n]['text'] for n in G2.nodes()]
    if not texts1 or not texts2:
        logger.warning("node_text_similarity: One of the graphs has no node texts.")
        return 0.0
    try:
        combined_text1 = ' '.join(texts1)
        combined_text2 = ' '.join(texts2)
        if len(combined_text1.strip()) < 3 or len(combined_text2.strip()) < 3:
            logger.warning("node_text_similarity: One of the texts is too short to compare.")
            return 0.0
        words1 = set(combined_text1.lower().split())
        words2 = set(combined_text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        return jaccard_sim
    except Exception:
        return 0.0


def graph_similarity(dict1, dict2, alpha=0.5):
    if not all(k in dict1 for k in ["ImplementationSteps", "ImplementationOrder"]) or \
            not all(k in dict2 for k in ["ImplementationSteps", "ImplementationOrder"]):
        logger.warning("graph_similarity: One of the graphs is missing necessary keys.")
        return 0.0
    if not dict1["ImplementationSteps"] or not dict1["ImplementationOrder"] or \
            not dict2["ImplementationSteps"] or not dict2["ImplementationOrder"]:
        logger.warning("graph_similarity: One of the graphs is missing necessary keys.")
        return 0.0
    try:
        G1 = nx.DiGraph()
        G2 = nx.DiGraph()
        for k, v in dict1["ImplementationSteps"].items():
            G1.add_node(str(k), text=v)
        for k, v in dict2["ImplementationSteps"].items():
            G2.add_node(str(k), text=v)
        if len(G1.nodes()) == 0 or len(G2.nodes()) == 0:
            return 0.0

        def process_order_items(order_list, graph, step_keys):
            edges_added = False
            if all(o.isdigit() for o in order_list):
                nodes = sorted([o for o in order_list if o in step_keys])
                for i in range(len(nodes) - 1):
                    graph.add_edge(nodes[i], nodes[i + 1])
                    edges_added = True
            else:
                for o in order_list:
                    if "-" in o:
                        try:
                            src, dst = o.split("-")
                            if src in step_keys and dst in step_keys:
                                graph.add_edge(src, dst)
                                edges_added = True
                        except Exception:
                            pass
            return edges_added

        step_keys1 = [str(k) for k in dict1["ImplementationSteps"].keys()]
        step_keys2 = [str(k) for k in dict2["ImplementationSteps"].keys()]
        edges_added_G1 = process_order_items(dict1["ImplementationOrder"], G1, step_keys1)
        edges_added_G2 = process_order_items(dict2["ImplementationOrder"], G2, step_keys2)
        if not edges_added_G1:
            nodes1 = sorted([n for n in G1.nodes()])
            for i in range(len(nodes1) - 1):
                G1.add_edge(nodes1[i], nodes1[i + 1])
                edges_added_G1 = True
        if not edges_added_G2:
            nodes2 = sorted([n for n in G2.nodes()])
            for i in range(len(nodes2) - 1):
                G2.add_edge(nodes2[i], nodes2[i + 1])
                edges_added_G2 = True
        if not edges_added_G1 or not edges_added_G2:
            logger.warning(
                "graph_similarity: One of the graphs has no edges, only node text similarity will be computed.")
            return node_text_similarity(G1, G2)
        edge_sim = edge_jaccard(G1, G2)
        text_sim = node_text_similarity(G1, G2)
        return alpha * edge_sim + (1 - alpha) * text_sim
    except Exception:
        return 0.0


def calculate_semantic_repetition(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?。！？]', text) if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 0.0
    try:
        if embedding_model is None:
            logger.warning("embedding_model is not available, cannot compute semantic repetition")
            return 0.0
        sentence_embeddings = embedding_model.encode(sentences)
        similarity_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)
        upper_triangle = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                upper_triangle.append(similarity_matrix[i][j])
        if not upper_triangle:
            return 0.0
        avg_similarity = np.mean(upper_triangle)
        penalty = max(0, (avg_similarity - 0.2) * 10)
        return min(penalty, 10.0)
    except Exception as e:
        logger.error(f"calculate_semantic_repetition error: {e}")
        return 0.0


def get_vote_from_model(model, original_idea_data, generated_idea_data, context=None, swap_positions=False):
    original_idea_text = format_idea_data(original_idea_data)
    generated_idea_text = format_idea_data(generated_idea_data)

    # determine positions for evaluation
    if swap_positions:
        # swap positions: generated idea as A, original idea as B
        prompt = get_evaluation_prompt_modified(generated_idea_text, original_idea_text, context)
        positions_swapped = True
    else:
        # default positions: original idea as A, generated idea as B
        prompt = get_evaluation_prompt_modified(original_idea_text, generated_idea_text, context)
        positions_swapped = False

    MAX_RETRIES = 5
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = model.generate(message=dict(type='text', value=prompt), temperature=0.1)
            if response is None:
                retry_count += 1
                logger.warning(f"model {model.model} API call failed, retry {retry_count}")
                time.sleep(1)
                continue
            evaluation_result = parse_evaluation_result(response)
            if evaluation_result is None:
                retry_count += 1
                logger.warning(f"model {model.model} evaluation result parse error, retry {retry_count}")
                time.sleep(1)
                continue
            if positions_swapped:
                evaluation_result = flip_evaluation_result(evaluation_result)
            return evaluation_result
        except Exception as e:
            retry_count += 1
            logger.error(f"model {model.model} evaluation error: {e}, try {retry_count}")
            time.sleep(1)

    logger.warning(f"model {model.model} evaluation failed after {MAX_RETRIES} retries")
    return None


def compare_ideas_with_voting(original_idea_data, generated_idea_data, context=None, judge_models=None):
    dimensions = ["effectiveness", "novelty", "detailedness", "feasibility", "overall"]
    vote_counts = {
        dim: {"original": 0, "generated": 0} for dim in dimensions
    }
    all_evaluations = []

    for model in judge_models:
        for swap in [False, True]:  # each model votes twice, once with normal positions, once with swapped positions
            evaluation = get_vote_from_model(
                model=model,
                original_idea_data=original_idea_data,
                generated_idea_data=generated_idea_data,
                context=context,
                swap_positions=swap
            )
            if evaluation:
                vote_detail = {
                    "model": model,
                    "positions_swapped": swap,
                    "results": {}
                }
                for dim in dimensions:
                    dim_result = evaluation.get(dim, {})
                    judgment = dim_result.get("judgment", "")
                    reason = dim_result.get("reason", "No reason provided")
                    if judgment == "win_A":
                        vote_counts[dim]["original"] += 1
                        result = "original_wins"
                    elif judgment == "win_B":
                        vote_counts[dim]["generated"] += 1
                        result = "generated_wins"
                    else:
                        logger.warning(f"error: {judgment}")
                        continue
                    vote_detail["results"][dim] = {
                        "result": result,
                        "reason": reason
                    }
                all_evaluations.append(vote_detail)
            else:
                logger.error(f"model {model} evaluation failed, could not get votes")

    final_results = {}
    for dim in dimensions:
        original_votes = vote_counts[dim]["original"]
        generated_votes = vote_counts[dim]["generated"]
        lose_gate = 2
        if dim == "novelty":
            win_gate = 4
        else:
            win_gate = 3

        if generated_votes > win_gate:
            result = "win"
            reason = f"Generated idea received {generated_votes} votes, Original idea received {original_votes} votes."
        elif generated_votes <= lose_gate:
            result = "lose"
            reason = f"Original idea received {original_votes} votes, Generated idea received {generated_votes} votes."
        else:
            result = "tie"
            reason = f"Generated idea received {generated_votes} votes, Original idea received {original_votes} votes."

        final_results[dim] = {
            "res": result,
            "reason": reason,
            "vote_detail": {
                "original_votes": original_votes,
                "generated_votes": generated_votes
            }
        }

    return {
        "final_results": final_results,
        "all_evaluations": all_evaluations
    }


# ImprovedIdeaEvaluator class
class ImprovedIdeaEvaluator:
    def __init__(self, idea_dict: dict):
        self.idea_dict = idea_dict
        self.original_data = {k: v for k, v in idea_dict.items() if k not in ["generated_idea_text", "generated_data"]}
        self.original_data["Idea"] = self.original_data.get("core_idea", "")
        self.original_data["RelatedWork"] = ast.literal_eval(self.original_data.get("related_work", "{}"))
        self.original_data["ExistingSolutions"] = ast.literal_eval(self.original_data.get("existing_solutions", "{}"))
        self.original_data["ImplementationSteps"] = ast.literal_eval(
            self.original_data.get("implementation_steps", "{}"))
        self.original_data["ImplementationOrder"] = ast.literal_eval(
            self.original_data.get("implementation_order", "[]"))
        self.original_data["EvaluationMetrics"] = ast.literal_eval(self.original_data.get("evaluation_metrics", "{}"))
        self.original_data["Dataset"] = self.original_data.get("data", "")
        self.original_data["ExpectedOutcome"] = self.original_data.get("expected_outcome", "")
        self.generated_data = idea_dict["generated_data"]
        self.idea = self.generated_data.get("Idea", "")
        self.generated_data["Idea"] = self.idea
        self.implementation_steps = self.generated_data.get("ImplementationSteps", {})
        self.implementation_order = self.generated_data.get("ImplementationOrder", {})
        self.dataset = self.generated_data.get("Dataset", "")
        self.generated_data["Dataset"] = self.dataset
        self.evaluation_metrics = self.generated_data.get("EvaluationMetrics", "")
        self.expected_outcome = self.generated_data.get("ExpectedOutcome", "")
        self.raw_scores = {
            "novelty_similarity": 0.0,
            "cutting_edge": 0.0,
            "effectiveness_objective": 0.0,
            "feasibility_objective": 0.0,
            "completeness": 0.0,
            "length_penalty": 0.0,
            "repetition_penalty": 0.0
        }
        self.scores = {
            "novelty_objective": 0.0,
            "feasibility_objective": 0.0,
            "detailedness_objective": 0.0,
            "effectiveness_objective": 0.0,
            "novelty": "",
            "effectiveness": "",
            "detailedness": "",
            "feasibility": "",
        }
        self.details = {}

    def evaluate_novelty_objective(self) -> None:
        try:
            text_to_compare = self.idea
            related_work = self.original_data.get("RelatedWork", {})
            existing_methods = self.original_data.get("ExistingSolutions", {})
            all_existing_text = []
            all_existing_text.extend(related_work.values())
            all_existing_text.extend(existing_methods.values())
            if all_existing_text and embedding_model is not None:
                idea_embedding = embedding_model.encode([text_to_compare])
                similarities = []
                for existing_text in all_existing_text:
                    existing_embedding = embedding_model.encode([existing_text])
                    similarity = cosine_similarity(
                        idea_embedding.reshape(1, -1),
                        existing_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(similarity)
                avg_similarity = np.mean(similarities)
                novelty_similarity_score = (1 - avg_similarity) * 10
                novelty_similarity_score = max(0, min(10, novelty_similarity_score))
            else:
                novelty_similarity_score = 0.0
            self.raw_scores["novelty_similarity"] = novelty_similarity_score
            ref_related_work = self.original_data.get("related_work_test", "")
            idea_embedding = embedding_model.encode([self.idea])
            similarities = []
            ref_related_work = ast.literal_eval(ref_related_work)
            for key, value in ref_related_work.items():
                snippet_data = f"{key}: {value}"
                snippet_embedding = embedding_model.encode([snippet_data])
                similarity = cosine_similarity(
                    idea_embedding.reshape(1, -1),
                    snippet_embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)
            avg_similarity = np.mean(similarities)
            cutting_edge_score = (1 - avg_similarity) * 10
            cutting_edge_score = max(0, min(10, cutting_edge_score))
            self.raw_scores["cutting_edge"] = cutting_edge_score
        except Exception as e:
            logger.error(f"Error in novelty evaluation: {e}")
            self.raw_scores["novelty_similarity"] = 0.0
            self.raw_scores["cutting_edge"] = 0.0
            self.details["novelty_similarity"] = f"error: {str(e)}"
            self.details["cutting_edge"] = f"error: {str(e)}"

    def evaluate_effectiveness_objective(self) -> None:
        try:
            original_terms = self.original_data.get("keywords", [])
            if embedding_model is None:
                self.scores["effectiveness_objective"] = 0.0
                self.details["effectiveness_objective"] = "embedding_model is not available"
                return
            terms_text = ", ".join([str(term) for term in original_terms])
            idea_text = self.idea
            try:
                embeddings = embedding_model.encode([terms_text, idea_text], normalize_embeddings=True)
                similarity = np.dot(embeddings[0], embeddings[1])
                prof_score = similarity * 10
                self.scores["effectiveness_objective"] = max(0, min(10, prof_score))
            except Exception as e:
                logger.error(f"Error computing embedding similarity: {e}")
                matched_terms = []
                generated_text_lower = idea_text.lower() if isinstance(idea_text, str) else ""
                for term in original_terms:
                    term_str = str(term).lower()
                    if term_str in generated_text_lower:
                        matched_terms.append(term)
                hit_rate = len(matched_terms) / len(original_terms) if original_terms else 0
                self.scores["effectiveness_objective"] = hit_rate * 10
                similarity = hit_rate
        except Exception as e:
            logger.error(f"Error in effectiveness_objective evaluation: {e}")
            self.scores["effectiveness_objective"] = 0.0

    def evaluate_completeness(self) -> None:
        required_sections = [
            "Idea",
            "ImplementationSteps",
            "ImplementationOrder",
            "EvaluationMetrics",
            "Dataset",
            "ExpectedOutcome"
        ]
        section_found = {
            "Idea": self.idea is not None,
            "ImplementationSteps": self.implementation_steps is not None,
            "ImplementationOrder": self.implementation_order is not None,
            "EvaluationMetrics": self.evaluation_metrics is not None,
            "Data": self.dataset is not None,
            "ExpectedOutcome": self.expected_outcome is not None
        }
        total_sections = len(required_sections)
        completed_sections = sum(section_found.values())
        self.raw_scores["completeness"] = (completed_sections / total_sections) * 10
        self.details["completeness"] = {
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "completion_rate": completed_sections / total_sections,
        }
        missing_sections = [section for section, found in section_found.items() if not found]
        if missing_sections:
            logger.warning(f"Missing required sections: {', '.join(missing_sections)}")

    def evaluate_feasibility_objective(self) -> None:
        try:
            generated_implementation = {
                "ImplementationSteps": self.implementation_steps,
                "ImplementationOrder": self.implementation_order
            }
            original_implementation = {
                "ImplementationSteps": self.original_data["ImplementationSteps"],
                "ImplementationOrder": self.original_data["ImplementationOrder"]
            }
            similarity = graph_similarity(
                generated_implementation,
                original_implementation,
                alpha=0.6
            )
            self.scores["feasibility_objective"] = similarity * 10
            self.details["feasibility_objective"] = {
                "score": similarity,
            }
        except Exception as e:
            logger.error(f"Error evaluating feasibility objective: {e}")
            self.scores["feasibility_objective"] = 0.0
            self.details["feasibility_objective"] = {"error": str(e)}

    def evaluate_penalties(self) -> None:
        if self.idea:
            char_count = len(self.idea)
            penalty = 0.0
            if char_count > 700:
                excess_chars = char_count - 700
                penalty += excess_chars / 100.0
            elif char_count < 300:
                deficit_chars = 300 - char_count
                penalty += deficit_chars / 100.0
            self.raw_scores["length_penalty"] = min(penalty, 10.0)
        else:
            self.raw_scores["length_penalty"] = 0.0
        if isinstance(self.idea, str):
            self.raw_scores["repetition_penalty"] = calculate_semantic_repetition(self.idea)
        else:
            self.raw_scores["repetition_penalty"] = 0.0
        self.details["penalties"] = {
            "text_length": len(self.idea),
            "length_penalty": self.raw_scores["length_penalty"],
            "repetition_penalty": self.raw_scores["repetition_penalty"]
        }

    def LLM_multi_rounds(self, llm_judges):
        try:
            context = get_context_from_data(self.original_data)
            evaluation_results = compare_ideas_with_voting(
                original_idea_data=self.original_data,
                generated_idea_data=self.generated_data,
                context=context,
                judge_models=llm_judges
            )
            summary = {
                "evaluation_details": evaluation_results,
                "timestamp": datetime.now().isoformat()
            }
            self.scores["novelty_subjective"] = evaluation_results["final_results"]["novelty"]["res"]
            self.scores["effectiveness_subjective"] = evaluation_results["final_results"]["effectiveness"]["res"]
            self.scores["detailedness_subjective"] = evaluation_results["final_results"]["detailedness"]["res"]
            self.scores["feasibility_subjective"] = evaluation_results["final_results"]["feasibility"]["res"]
            return {
                "success": True,
                "result": summary
            }
        except Exception as e:
            logger.error(f"Error in LLM_multi_rounds: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def merge_scores(self) -> None:
        self.scores["novelty_objective"] = (
            0.5 * self.raw_scores["novelty_similarity"]
            + 0.5 * self.raw_scores["cutting_edge"]
        )
        self.scores["detailedness_objective"] = (
            0.2 * self.raw_scores["completeness"]
            + 0.4 * (10 - self.raw_scores["repetition_penalty"])
            + 0.4 * (10 - self.raw_scores["length_penalty"])
        )

    def calculate_final_score(self, llm_judges) -> Dict[str, Any]:
        self.LLM_multi_rounds(llm_judges)
        self.evaluate_novelty_objective()
        self.evaluate_effectiveness_objective()
        self.evaluate_completeness()
        self.evaluate_feasibility_objective()
        self.evaluate_penalties()
        self.merge_scores()

        self.idea_dict.update({
            "effectiveness_objective": float(self.scores["effectiveness_objective"]) * 10,
            "novelty_objective": float(self.scores["novelty_objective"]) * 10,
            "detailedness_objective": float(self.scores["detailedness_objective"]) * 10,
            "feasibility_objective": float(self.scores["feasibility_objective"]) * 10,
            "effectiveness_subjective": 100 if self.scores["effectiveness_subjective"] == 'win' else 0,
            "novelty_subjective": 100 if self.scores["novelty_subjective"] == 'win' else 0,
            "detailedness_subjective": 100 if self.scores["detailedness_subjective"] == 'win' else 0,
            "feasibility_subjective": 100 if self.scores["feasibility_subjective"] == 'win' else 0,
        })

        self.idea_dict.update({
            "effectiveness": (self.idea_dict["effectiveness_objective"] + self.idea_dict[
                "effectiveness_subjective"]) / 2,
            "novelty": (self.idea_dict["novelty_objective"] + self.idea_dict["novelty_subjective"]) / 2,
            "detailedness": (self.idea_dict["detailedness_objective"] + self.idea_dict["detailedness_subjective"]) / 2,
            "feasibility": (self.idea_dict["feasibility_objective"] + self.idea_dict["feasibility_subjective"]) / 2,
        })

        self.idea_dict["final_score"] = (
            self.idea_dict["effectiveness"]
            + self.idea_dict["novelty"]
            + self.idea_dict["detailedness"]
            + self.idea_dict["feasibility"]
        ) / 4

        return self.idea_dict


def evaluate_single_idea(ques_dict, llm_judges):
    try:
        evaluator = ImprovedIdeaEvaluator(ques_dict)
        evaluation_result = evaluator.calculate_final_score(llm_judges=llm_judges)
        output = evaluation_result
        return output
    except Exception as e:
        logger.error(f"evaluation error: {e}")
        output = {
            "error": str(e),
            "final_score": 0.0
        }
        return output


class SGI_Bench_Idea_Generation(TextBaseDataset):
    TYPE = 'QA'
    example = {
        "Idea": (
            "We propose an adaptive optimization framework based on a dynamic feature interaction "
            "network. This framework captures feature correlations through a hierarchical attention "
            "mechanism and combines it with a data distribution-aware dynamic weight adjustment "
            "strategy to improve the model's adaptability to heterogeneous data while ensuring "
            "computational efficiency."
        ),
        "ImplementationSteps": {
            "1": (
                "Data preprocessing: missing value filling, outlier handling, feature "
                "normalization and type conversion, and building a basic feature set"
            ),
            "2": (
                "Feature engineering: generating statistically derived features, time series "
                "features, and cross-features, and building a feature candidate pool"
            ),
            "3": (
                "Model architecture design: building a basic network module, integrating a "
                "hierarchical attention mechanism with a dynamic interaction layer"
            ),
            "4": (
                "Dynamic weight mechanism implementation: designing a data distribution-aware "
                "weight adjustment function and embedding it into the network's intermediate layers"
            ),
            "5": (
                "Model training and tuning: adopting a phased training strategy, using grid "
                "search and early stopping to optimize hyperparameters"
            ),
            "6": (
                "Performance Verification: Conduct comparative experiments on multiple datasets "
                "to analyze model performance differences in different scenarios."
            )
        },
        "ImplementationOrder": ["1-2", "2-3", "3-4", "4-5", "1-5", "5-6"],
        "Dataset": (
            "Contains three types of public datasets and one actual business data: "
            "1) Public structured dataset (approximately 500,000 samples, 30+ features); "
            "2) Text-numeric mixed dataset (approximately 200,000 samples, including text "
            "embedding features); 3) Time series sparse dataset (approximately 100,000 samples, "
            "spanning 1 year); 4) Real transaction data from an e-commerce platform "
            "(approximately 1 million samples, including user behavior and product attribute "
            "features)"
        ),
        "EvaluationMetrics": {
            "Prediction Accuracy": (
                "AUC and F1-score are used for classification tasks; MAE and RMSE are used for "
                "regression tasks to evaluate the basic predictive ability of the model."
            ),
            "Robustness": (
                "Performance decay rate is calculated through data perturbation testing (adding "
                "noise and simulating feature loss) to measure model stability."
            ),
            "Efficiency": (
                "Record model training time, inference latency, and memory usage to evaluate "
                "computing resource consumption."
            ),
            "Interpretability": (
                "Use SHAP values and feature importance ranking to quantify the feature "
                "contribution to model decisions."
            ),
            "Generalization": (
                "Performance retention across datasets to evaluate the model's adaptability "
                "to unseen data."
            )
        },
        "ExpectedOutcome": (
            "The proposed framework outperforms existing mainstream methods in comprehensive "
            "performance (accuracy, robustness, and efficiency) across multiple datasets, "
            "particularly in scenarios with uneven data distribution and cross-scenario migration. "
            "It also enhances model interpretability through a dynamic feature interaction "
            "mechanism, providing effective support for practical business decision-making."
        )
    }

    @classmethod
    def supported_datasets(cls):
        return ["SGI-IdeaGeneration"]

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-IdeaGeneration", split="test")
        rows: List[Dict[str, Any]] = []
        idx = 0
        for prob in hf:
            rows.append({
                "index": idx,
                "id": prob.get("idx", idx),
                "question": prob["question"],
                "discipline": prob["discipline"],
                "core_idea": prob["core_idea"],
                "related_work": prob["related_work"],
                "related_work_test": prob.get("related_work_test", "{}"),
                "existing_solutions": prob["existing_solutions"],
                "implementation_steps": prob["implementation_steps"],
                "implementation_order": prob["implementation_order"],
                "data": prob["data"],
                "evaluation_metrics": prob["evaluation_metrics"],
                "expected_outcome": prob["expected_outcome"],
                "keywords": prob.get("keywords", [])
            })
            idx += 1
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        prompt = line['question'] + f"""\n\n### Example:
```json
{json.dumps(self.example, indent=4)}
```"""
        msgs = [{'type': 'text', 'value': prompt}]
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = pd.DataFrame(data)
        global embedding_model
        # 尝试加载嵌入模型进行评估
        if embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer embedding model...")
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                embedding_model = None

        data['generated_data'] = None
        data['generated_data'] = data['generated_data'].astype(object)

        data['generated_idea_text'] = None
        data['generated_idea_text'] = data['generated_idea_text'].astype(object)

        # 处理每个生成的想法
        # Default judge models
        JUDGE_MODELS = ["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5-20250929"]
        # JUDGE_MODELS = ["gpt-5.1-2025-11-13", "gemini-3-pro-preview", "claude-sonnet-4-5-20250929"]
        llm_judges = [build_judge(**{**judge_kwargs, 'model': i}) for i in JUDGE_MODELS]
        tups = []
        indices = []
        for idx, row in data.iterrows():
            prediction = row['prediction']

            # 解析生成的想法
            if isinstance(prediction, str):
                parsed_data = parse_generated_idea(prediction)
                data.at[idx, 'generated_data'] = parsed_data
                data.at[idx, 'generated_idea_text'] = prediction

                # 构建评估所需的原始数据字典
                ques_dict = row.to_dict()
                ques_dict['generated_data'] = parsed_data
                ques_dict['generated_idea_text'] = prediction

                # 评估单个想法
                tups.append((ques_dict, llm_judges))
                indices.append(idx)

        tmp_file = get_intermediate_file_path(eval_file, '_judge_tmp', 'pkl')
        ans = {}
        if osp.exists(tmp_file):
            ans = load(tmp_file)
        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]

        if len(indices):
            track_progress_rich(
                func=evaluate_single_idea,
                tasks=tups,
                nproc=judge_kwargs.get('nproc', 32),
                save=tmp_file,
                keys=indices,
            )
            ans = load(tmp_file)

        for idx, evaluation_result in ans.items():
            # 将评估结果添加到数据中
            for key, value in evaluation_result.items():
                if key not in ['generated_data', 'generated_idea_text']:
                    data.loc[idx, key] = value

        # 计算平均分数
        successful_evaluations = data[~data['final_score'].isna()]
        if len(successful_evaluations) > 0:
            avg_effectiveness_objective = successful_evaluations['effectiveness_objective'].mean()
            avg_novelty_objective = successful_evaluations['novelty_objective'].mean()
            avg_detailedness_objective = successful_evaluations['detailedness_objective'].mean()
            avg_feasibility_objective = successful_evaluations['feasibility_objective'].mean()

            avg_effectiveness_subjective = successful_evaluations['effectiveness_subjective'].mean()
            avg_novelty_subjective = successful_evaluations['novelty_subjective'].mean()
            avg_detailedness_subjective = successful_evaluations['detailedness_subjective'].mean()
            avg_feasibility_subjective = successful_evaluations['feasibility_subjective'].mean()

            effectiveness_score = successful_evaluations['effectiveness'].mean()
            novelty_score = successful_evaluations['novelty'].mean()
            detailedness_score = successful_evaluations['detailedness'].mean()
            feasibility_score = successful_evaluations['feasibility'].mean()

            avg_final_score = successful_evaluations['final_score'].mean()

            result = {
                "final_score": float(avg_final_score),
                "effectiveness": float(effectiveness_score),
                "novelty": float(novelty_score),
                "detailedness": float(detailedness_score),
                "feasibility": float(feasibility_score),
                "details": {
                    "effectiveness_objective": float(avg_effectiveness_objective),
                    "effectiveness_subjective": float(avg_effectiveness_subjective),
                    "novelty_objective": float(avg_novelty_objective),
                    "novelty_subjective": float(avg_novelty_subjective),
                    "detailedness_objective": float(avg_detailedness_objective),
                    "detailedness_subjective": float(avg_detailedness_subjective),
                    "feasibility_objective": float(avg_feasibility_objective),
                    "feasibility_subjective": float(avg_feasibility_subjective),
                    "successful_evaluations": len(successful_evaluations),
                    "total_evaluations": len(data)
                }
            }
        else:
            result = {
                "final_score": 0.0,
                "effectiveness": 0.0,
                "novelty": 0.0,
                "detailedness": 0.0,
                "feasibility": 0.0,
                "error": "No successful evaluations"
            }

        # 保存结果
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(data, score_file)
        dump(result, result_file)

        return result
