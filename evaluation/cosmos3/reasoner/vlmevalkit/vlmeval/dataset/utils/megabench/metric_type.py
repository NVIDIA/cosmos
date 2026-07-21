import logging
from enum import Enum
from functools import cached_property

from .utils import lazy_import


class MetricType(Enum):
    """The types of metrics."""

    EXACT_STR_MATCH = "exact_str_match"
    SIMPLE_STR_MATCH = "simple_str_match"
    CODE_RESULT_EXACT_STR_MATCH = "code_result_exact_str_match"
    DICT_EXACT_STR_MATCH_AGG_RECALL = "dict_exact_str_match_agg_recall"
    EXACT_STR_MATCH_CASE_INSENSITIVE = "exact_str_match_case_insensitive"
    NORM_SIM_DAMERAU_LEVENSHTEIN = "normalized_similarity_damerau_levenshtein"
    NEAR_STR_MATCH = "near_str_match"
    NUMBER_RELATIVE_DIFF_RATIO = "number_rel_diff_ratio"
    SET_EQUALITY = "set_equality"
    SET_EQUALITY_CASE_INSENSITIVE = "set_equality_case_insensitive"
    DICT_SET_EQUALITY_AGG_JACCARD = "dict_set_equality_agg_jaccard"
    DICT_PRECISION = "dict_precision"
    JACCARD_INDEX = "jaccard_index"
    JACCARD_INDEX_CASE_INSENSITIVE = "jaccard_index_case_insensitive"
    DICT_JACCARD_AGG_JACCARD = "dict_jaccard_agg_jaccard"
    DICT_EQUALITY = "dict_equality"
    SET_PRECISION = "set_precision"
    POSITIVE_INT_MATCH = "positive_int_match"
    CHESS_MOVE_LIST_JACCARD_INDEX = "chess_move_list_jaccard_index"
    LONGEST_COMMON_LIST_PREFIX_RATIO = "longest_common_list_prefix_ratio"
    ASCII_ART_GPT4O_JUDGE = "ascii_art_gpt4o_judge"
    NLI_ENTAILMENT = "nli_entailment"
    BLEU = "bleu"
    GLEU_CN = "gleu_cn"
    XML_NORM_BBOX_IOU_SINGLE = "xml_nbbox_iou_single"
    LATEX_EXPR_EQUALITY = "latex_expr_equality"
    TEXT_WITH_LATEX_EXPR_EQUALITY = "text_with_latex_expr_equality"
    NORM_BBOX_IOU_TUPLE = "nbbox_iou_tuple"
    NORM_BBOX_IOU_SINGLE = "nbbox_iou_single"
    NORM_BBOX_IOU_SEQUENCE = "nbbox_iou_sequence"
    DICT_NORM_BBOX_IOU_TUPLE_AGG_JACCARD = "dict_nbbox_iou_tuple_agg_jaccard"
    XML_NORM_POINT_IN_BBOX = "xml_norm_point_in_bbox"
    XML_NORM_POINT_DISTANCE = "xml_norm_point_distance"
    GEO_PROXIMITY_LOCATION_DICT = "geo_proximity_location_dict"
    NORMALIZED_RMSE = "normalized_rmse"
    PROGRAM_JUDGE = "program_judge"
    STR_SET_EQUALITY_LINE_BREAK = "str_set_equality_line_break"
    STR_SET_EQUALITY_COMMA = "str_set_equality_comma"
    SEQUENCE_EQUALITY = "sequence_equality"
    SEQUENCE_EQUALITY_CASE_INSENSITIVE = "sequence_equality_case_insensitive"
    SEQUENCE_ACCURACY_CASE_INSENSITIVE = "sequence_accuracy_case_insensitive"
    ANGLE_SEQ_FLOAT_RMSE = "angle_seq_float_rmse"
    SYMBOLIC_PLANNING_TEST = "symbolic_planning_test"
    MULTI_REF_PHRASE_EVAL = "multi_ref_phrase"
    GENERAL_SINGLE_NUMERICAL_MATCH = "general_single_numerical_match"
    BOXED_SINGLE_NUMERICAL_MATCH = "boxed_single_numerical_match"
    SEQUENCE_COORDS_SIMILARITY = "sequence_coords_similarity"
    CONSTRAINED_GENERATION = "constrained_generation"
    VLM_AS_JUDGE = "gpt_4o_as_judge"
    UNSUPPORTED = "unsupported"

    @cached_property
    def class_impl(self):
        lazy_imports = {
            MetricType.SIMPLE_STR_MATCH: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.simple_str_match", "SimpleStrMatch"
            ),
            MetricType.EXACT_STR_MATCH: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.exact_str_match", "ExactStrMatch"
            ),
            MetricType.CODE_RESULT_EXACT_STR_MATCH: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.exact_str_match", "CodeResultExactStrMatch"
            ),
            MetricType.DICT_EXACT_STR_MATCH_AGG_RECALL: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.dict_exact_match_agg_recall",
                "DictExactStrMatchAggRecall",
            ),
            MetricType.EXACT_STR_MATCH_CASE_INSENSITIVE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.exact_str_match_case_insensitive",
                "ExactStrMatchCaseInsensitive",
            ),
            MetricType.NORM_SIM_DAMERAU_LEVENSHTEIN: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.normalized_similarity_damerau_levenshtein",
                "NormalizedSimilarityDamerauLevenshtein",
            ),
            MetricType.NEAR_STR_MATCH: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.near_str_match", "NearStrMatch"
            ),
            MetricType.NUMBER_RELATIVE_DIFF_RATIO: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.number_rel_diff_ratio", "NumberRelDiffRatio"
            ),
            MetricType.SET_EQUALITY: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.set_equality", "SetEquality"
            ),
            MetricType.SET_EQUALITY_CASE_INSENSITIVE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.set_equality", "SetEqualityCaseInsensitive"
            ),
            MetricType.DICT_SET_EQUALITY_AGG_JACCARD: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.dict_set_equality_agg_jaccard",
                "DictSetEqualityAggJaccard",
            ),
            MetricType.DICT_EQUALITY: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.dict_equality",
                "DictEquality",
            ),
            MetricType.DICT_PRECISION: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.dict_equality",
                "DictPrecision",
            ),
            MetricType.JACCARD_INDEX: lazy_import("vlmeval.dataset.utils.megabench.scoring.jaccard", "Jaccard"),
            MetricType.JACCARD_INDEX_CASE_INSENSITIVE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.jaccard", "JaccardCaseInsensitive"
            ),
            MetricType.DICT_JACCARD_AGG_JACCARD: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.dict_jaccard_agg_jaccard", "DictJaccardAggJaccard"
            ),
            MetricType.SET_PRECISION: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.set_precision", "SetPrecision"
            ),
            MetricType.POSITIVE_INT_MATCH: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.positive_int_match", "PositiveIntMatch"
            ),
            MetricType.CHESS_MOVE_LIST_JACCARD_INDEX: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.chess_jaccard", "ChessMoveJaccard"
            ),
            MetricType.LONGEST_COMMON_LIST_PREFIX_RATIO: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.longest_common_list_prefix_ratio",
                "LongestCommonListPrefixRatio",
            ),
            MetricType.ASCII_ART_GPT4O_JUDGE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.ascii_art_gpt4o_judge",
                "AsciiArtVLMJudgeScore",
            ),
            MetricType.NLI_ENTAILMENT: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.nli_entailment", "NliEntailment"
            ),
            MetricType.BLEU: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.sacrebleu_bleu",
                "Bleu",
            ),
            MetricType.GLEU_CN: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.gleu",
                "GLEUChinese",
            ),
            MetricType.XML_NORM_BBOX_IOU_SINGLE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.xml_nbbox_iou", "XmlNbboxIouSingle"
            ),
            MetricType.BOXED_SINGLE_NUMERICAL_MATCH: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.general_numerical_match", "BoxedSingleNumericalMatch"
            ),
            MetricType.GENERAL_SINGLE_NUMERICAL_MATCH: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.general_numerical_match", "GeneralSingleNumericalMatch"
            ),
            MetricType.SEQUENCE_COORDS_SIMILARITY: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.coordinate_sequence_match", "CoordsSequenceSimilarity"
            ),
            MetricType.LATEX_EXPR_EQUALITY: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.latex_expr_equality",
                "LatexExprEquality",
            ),
            MetricType.TEXT_WITH_LATEX_EXPR_EQUALITY: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.latex_expr_equality",
                "TextLatexExprEquality",
            ),
            MetricType.NORM_BBOX_IOU_TUPLE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.nbbox_iou", "NbboxIouTuple"
            ),
            MetricType.NORM_BBOX_IOU_SINGLE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.nbbox_iou", "NbboxIouSingle"
            ),
            MetricType.NORM_BBOX_IOU_SEQUENCE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.nbbox_iou", "NbboxIouSequence"
            ),
            MetricType.DICT_NORM_BBOX_IOU_TUPLE_AGG_JACCARD: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.dict_nbbox_iou_tuple_agg_jaccard",
                "DictNbboxIouTupleAggJaccard",
            ),
            MetricType.XML_NORM_POINT_IN_BBOX: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.xml_norm_point_in_bbox",
                "XmlNormPointInBbox",
            ),
            MetricType.XML_NORM_POINT_DISTANCE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.xml_norm_point_distance",
                "XmlNormPointDistance",
            ),
            MetricType.GEO_PROXIMITY_LOCATION_DICT: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.geo_proximity", "GeoProximityLocationDict"
            ),
            MetricType.NORMALIZED_RMSE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.mse", "NormalizedRMSE"
            ),
            MetricType.PROGRAM_JUDGE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.program_judge", "ProgramJudge"
            ),
            MetricType.STR_SET_EQUALITY_LINE_BREAK: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.set_equality", "StringSetEqualityLineSplit"
            ),
            MetricType.STR_SET_EQUALITY_COMMA: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.set_equality", "StringSetEqualityCommaSplit"
            ),
            MetricType.SEQUENCE_EQUALITY: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.sequence_equality", "SequenceEquality"
            ),
            MetricType.SEQUENCE_EQUALITY_CASE_INSENSITIVE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.sequence_equality", "SequenceEqualityCaseInsensitive"
            ),
            MetricType.SEQUENCE_ACCURACY_CASE_INSENSITIVE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.sequence_equality", "SequenceAccuracyCaseInsensitive"
            ),
            MetricType.ANGLE_SEQ_FLOAT_RMSE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.mse", "AngleSeqFloatRMSE"
            ),
            MetricType.SYMBOLIC_PLANNING_TEST: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.symbolic_planning", "SymbolicPlanningMetricTest"
            ),
            MetricType.MULTI_REF_PHRASE_EVAL: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.multi_ref_phrase", "MultipleReferencePhraseEval"
            ),
            MetricType.CONSTRAINED_GENERATION: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.constrained_generation", "ConstrainedGenerationEval"
            ),
            MetricType.VLM_AS_JUDGE: lazy_import(
                "vlmeval.dataset.utils.megabench.scoring.vlm_as_judge", "VLMJudgeScore"
            ),
        }

        if self not in lazy_imports:
            logging.error(f"Metric {self} not implemented...")

        importer = lazy_imports.get(
            self,
            lazy_import("vlmeval.dataset.utils.megabench.scoring.unsupported_scoring", "UnsupportedScoring"),
        )
        return importer()

    def match(self, response: str, correct_answer: str, task_info=None):
        if not task_info:
            return self.class_impl.match(response, correct_answer)
        else:
            return self.class_impl.match(response, correct_answer, task_info)

    @classmethod
    def from_string(cls, s):
        try:
            if s is None:
                return cls("unsupported")
            return cls(s.lower())
        except KeyError as exc:
            raise ValueError(f"Invalid metric type: {s}") from exc

    @classmethod
    def get_all_values(cls):
        return list(cls)


# List all of the supported metrics:
if __name__ == "__main__":
    print("All MetricType values:")
    for metric_type in MetricType.get_all_values():
        print(f"{metric_type.name}: {metric_type.value}")
