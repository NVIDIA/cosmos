import copy
import os.path as osp
import warnings

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, localize_df, toliststr
from .aetcbench import AETCBench
from .asclepius import Asclepius
# Fork-specific datasets
from .av_3d_grounding import ThreeDAVGroundingBench
from .av_prompt_following import AVPromptFollowingBench
from .av_speakerbench import AVSpeakerBench
from .avspecial_collision_bench import AVSpecialCollisionBench
from .avspecial_environment_bench import AVSpecialEnvironmentBench
from .avspecial_ood_reasoning_bench import AVSpecialOODReasoningBench
from .avspecial_stop_behavior_bench import AVSpecialStopBehaviorBench
from .blink_depth import BlinkDepth
from .blink_spatial import BlinkSpatial
from .camera_bench import CameraBench
from .camera_intrinsic import CameraIntrinsicBench
from .causalvqa import CausalVQA
from .CGAVCounting.cg_av_counting import CGAVCounting
from .cgbench import (CGBench_MCQ_Grounding, CGBench_MCQ_Grounding_Mini, CGBench_OpenEnded,
                      CGBench_OpenEnded_Mini)
from .chartbench import ChartBench
from .chartcap import ChartCapDataset
from .chartmimic import ChartMimic
from .chartmuseum import ChartMuseum
from .chartqapro import ChartQAPro
from .chartx import ChartX
from .charxiv import CharXiv
from .cmmmu import CMMMU
from .cosmos_cab_image import CosmosCABImage
from .cosmos_cab_video import CosmosCABVideoCamera, CosmosCABVideoGeneral
from .cosmos_erqa import CosmosERQA
from .worldbench import WorldBench
from .cosmos_reason import CosmosReason
from .creation import CreationMMBenchDataset
from .cv_bench import CVBench
from .da2k import DA2K
from .design2code import Design2Code
from .dream import DREAM
from .dsrbench import DSRBench
from .dude import DUDE
from .dynamath import Dynamath
from .EgoExoBench.egoexobench import EgoExoBench_MCQ
from .embspatialbench import EmbSpatialBench
from .emma import EMMADataset
from .eriq import ERIQBench
from .erqa import ERQADataset
from .erqabench import ERQABench
from .flames import FlamesDataset
from .foxbench import FoxBench
from .gobench import GOBenchDataset
from .groundingme import GroundingME
from .gsm8k_v import GSM8KVDataset
from .GUI.osworld_g import OSWorld_G
from .GUI.screenspot import ScreenSpot
from .GUI.screenspot_pro import ScreenSpot_Pro
from .GUI.screenspot_v2 import ScreenSpotV2
from .GUI.vbgd import VBGD
from .GUI.venusbench import VenusBench_GD
from .health_surgi_bench import HealthSurgiBench
from .hipho import HiPhODataset
from .IFBench.ifbench import IFBench
from .image_base import ImageBaseDataset, img_root_map
from .image_caption import ImageCaptionDataset
from .image_ccocr import CCOCRDataset
from .image_mcq import (CVQA, LEGO, SCAM, AffordanceDataset, CustomMCQDataset,
                        GMAIMMBenchDataset, HRBenchDataset, ImageMCQDataset, MedXpertQA_MM_test,
                        MicroBench, MMERealWorld, MMMUDataset, MMMUProDataset, MSEarthMCQ,
                        MUIRDataset, NaturalBenchDataset, OmniEarthMCQBench, OmniMedVQA, PuzzleVQA,
                        TDBench, TopViewRS, TreeBench, VisualPuzzles, VisuLogic, VLMBlind,
                        VMCBenchDataset, WeMath, XLRSBench, _3DSRBench)
from .image_mt import MMDUDataset
from .image_shortqa import ImageShortQADataset, PathVQA_TEST, PathVQA_VAL
from .image_vqa import (BMMR, CRPE, LENS, MMNIAH, AyaVisionBench, CoreCognition, CountBenchQA,
                        CustomVQADataset, ImageVQADataset, LLaVABench, LLaVABench_KO, LogicVista,
                        MathCanvas, MathVerse, MathVision, MathVista, MME_CoT, MMEReasoning,
                        MMReason, MMSci_Captioning, MMVet, MMVMBench, MTVQADataset, OCR_Reasoning,
                        OCRBench, OCRBench_v2, OlympiadBench, Omni3DBench, Physics_yale, PhyX,
                        QSpatial, SeePhys, TableVQABench, TallyQA, TDBenchGrounding, VGRPBench,
                        VizWiz, VLMsAreBiased, VTCBench, WildDocBenchmark, ZEROBench)
from .image_yorn import ImageYORNDataset
from .intphys2 import IntPhys2
from .its_collision import ITSCollision
from .lingoqa import LingoQA
from .locate_anything_bench import LocateAnythingBench
from .longvideobench import LongVideoBench
from .lv_event_verification import LVEventVerification
from .lvs import LVSDataset, LVSHallucinationDataset
from .lvs_ai_hallucination import LVSAIHallucinationDataset
from .m3oralbench import M3oralBenchDataset
from .m4bench import M4Bench
from .macbench import MaCBench
from .matbench import MATBench
from .medqbench_caption import MedqbenchCaptionDataset
from .medqbench_mcq import MedqbenchMCQDataset
from .medqbench_paired_description import MedqbenchPairedDescriptionDataset
from .megabench import MEGABench
from .metropolis2d.astro_2d_dataset import Astro2DDetectionDataset, _build_astro2dbench_dataset
from .metropolis2d.detection_2d_dataset import Metropolis2DDetectionDataset
from .metropolis2d.grounding_2d_dataset import Metropolis2DGroundingDataset
from .metropolis_dvc import MetropolisDVC
from .metropolis_event_verification import MetropolisEventVerification
from .metropolis_temporal import MetropolisTemporal
from .metropolis_vqa import MetropolisVQA
from .miabench import MIABench
from .mindcubebench import MindCubeBench
from .mlvu import MLVU, MLVU_MCQ, MLVU_OpenEnded
from .mmalignbench import MMAlignBench
from .mmbench_video import MMBenchVideo
from .mmesci import MMESCIDataset
from .mmgenbench import MMGenBench
from .mmhelix import MMHELIX
from .mmifeval import MMIFEval
from .mmlongbench import MMLongBench
from .mmmath import MMMath
from .mmoral_opg_closed import MMOral_OPG_CLOSED
from .mmoral_opg_open import MMOral_OPG_OPEN
from .mmsafetybench import MMSafetyBenchDataset
from .mmsibench import MMSIBench, MMSIVideoBench
from .moat import MOAT
from .moviechat1k import MovieChat1k
from .mssbench import MSSBenchDataset
from .mvbench import MVBench, MVBench_MP4
from .mvpbench import MVPBench
from .mvu_eval import MVUEval
from .NPMM import NPMM
from .oceanocr import OceanOCRBench
from .odinw13 import ODinW13Dataset
from .olmOCRBench.olmocrbench import olmOCRBench
from .Omni3D.omni3d_dataset import Omni3DDetectionDataset
from .OmniDocBench.omnidocbench import OmniDocBench
from .omnispatialbench import OmniSpatialBench
from .omtgbench import OMTGBench
from .ost_bench import OSTDataset
from .plotqa import PlotQA
from .qbench_video import QBench_Video, QBench_Video_MCQ, QBench_Video_VQA
from .reasonmap_plus import ReasonMap_Plus
from .refcoco import RefCOCODataset
from .refspatial import RefSpatialDataset
from .refspatialbench import RefSpatialBench
from .robospatial_home import RoboSpatialHome
from .robospatialbench import RoboSpatialBench
from .sarena import SArena
from .sat_bench import SATBench
from .sfebench import SFE
from .SGI_Bench_1_0.deep_research import SGI_Bench_Deep_Research
from .SGI_Bench_1_0.dry_experiment import SGI_Bench_Dry_Experiment
from .SGI_Bench_1_0.experimental_reasoning import SGI_Bench_Experimental_Reasoning
from .SGI_Bench_1_0.idea_generation import SGI_Bench_Idea_Generation
from .SGI_Bench_1_0.wet_experiment import SGI_Bench_Wet_Experiment
from .share_robot_bench import ShareRobotTrajectory
from .simplevqa import SimpleVQA
from .sitebench import SiteBenchImage, SiteBenchVideo
from .siuo import SIUODataset
from .siuo_gen import SIUOGenDataset
from .siuo_mcq import SIUOMCQDataset
from .slidevqa import SlideVQA
from .sop_action_recognition import SOPActionRecognition
from .sop_temporal_localization import SOPTemporalLocalization
from .spar_bench import SPARBench, SPARBenchTiny
from .sparbench import SparBench
from .spatial457 import Spatial457
from .spatialvizbench import SpatialVizBench
from .spbench import SPBench
from .ssi_bench import SSIBenchDataset
from .starebench import StareBench
from .stibench import STIBench
from .tailgating import TailgatingVerification
from .tamperbench import MVTamperBench
from .tempcompass import TempCompass, TempCompass_Captioning, TempCompass_MCQ, TempCompass_YorN
from .temporal_localization_bench import TemporalLocalization
from .text_mcq import CustomTextMCQDataset, TextMCQDataset
from .uni_svg import UniSVG
from .utils import DEBUG_MESSAGE, build_judge, extract_answer_from_item, prefetch_answer
from .v2pbench import V2PBench
from .vantage_pointing import VANTAGE_2DPointing
from .vantage_sot import VANTAGE_SOT
from .vcr import VCRDataset
from .vcrbench import VCRBench
from .vdc import VDC
from .video_concat_dataset import ConcatVideoDataset
from .video_holmes import Video_Holmes
from .motionbench import MotionBench
from .video_mmlu import Video_MMLU_CAP, Video_MMLU_QA
from .videomme import VideoMME
from .videommmu import VideoMMMU
from .videophy2 import VideoPhy2
from .videott import VideoTT
from .viewspatialbench import ViewSpatialBench
from .visfactor import VisFactor
from .vl_rewardbench import VLRewardBench
from .vladbench import VLADBench
from .vlm2bench import VLM2Bench
from .vlmbias import VLMBias
from .vlrmbench import VLRMBench
from .vsibench import VsiBench, VsiSuperCount, VsiSuperRecall
from .vss_oobe_clips import VSSoobeClips
from .warehouse_near_miss import WarehouseNearMiss
from .warehouse_spatial_ai import WarehouseSpatialAI
from .where2place import Where2Place
from .wildvision import WildVision
from .worldsense import WorldSense
from .worldvqa import WorldVQA
from .xstest import XSTestDataset

from .video_dataset_config import supported_video_datasets  # isort: skip


class ConcatDataset(ImageBaseDataset):
    # This dataset takes multiple dataset names as input and aggregate them into a single dataset.
    # Each single dataset should not have a field named `SUB_DATASET`

    DATASET_SETS = {
        'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
        'MTL_MMBench_DEV': [
            'MMBench_dev_ar', 'MMBench_dev_cn', 'MMBench_dev_en',
            'MMBench_dev_pt', 'MMBench_dev_ru', 'MMBench_dev_tr'
        ],
        'ScreenSpot_Pro': [
            'ScreenSpot_Pro_Development', 'ScreenSpot_Pro_Creative', 'ScreenSpot_Pro_CAD',
            'ScreenSpot_Pro_Scientific', 'ScreenSpot_Pro_Office', 'ScreenSpot_Pro_OS'
        ],
        'ScreenSpot': ['ScreenSpot_Mobile', 'ScreenSpot_Desktop', 'ScreenSpot_Web'],
        'ScreenSpot_v2': ['ScreenSpot_v2_Mobile', 'ScreenSpot_v2_Desktop', 'ScreenSpot_v2_Web'],
        'M4Bench': ['State_Invariance', 'State_Comparison', 'Spatial_Perception', 'Instance_Comparison', 'Detailed_Difference'],  # noqa: E501
    }

    def __init__(self, dataset, **kwargs):
        # Forward profile-level kwargs (e.g. `model_family` from `--profile`)
        # to each child so concat-wrapped benchmarks can per-family dispatch.
        # Children that don't consume a given kwarg absorb it via
        # ImageBaseDataset's `**kwargs` catch-all.
        datasets = self.DATASET_SETS[dataset]
        self.dataset_map = {}
        # The name of the compliation
        self.dataset_name = dataset
        self.datasets = datasets
        for dname in datasets:
            dataset = build_dataset(dname, **kwargs)
            assert dataset is not None, dataset
            self.dataset_map[dname] = dataset
        TYPES = [x.TYPE for x in self.dataset_map.values()]
        MODALITIES = [x.MODALITY for x in self.dataset_map.values()]
        assert np.all([x == TYPES[0] for x in TYPES]), (datasets, TYPES)
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (datasets, MODALITIES)
        self.TYPE = TYPES[0]
        self.MODALITY = MODALITIES[0]
        data_all = []
        for dname in datasets:
            data = self.dataset_map[dname].data
            data['SUB_DATASET'] = [dname] * len(data)
            if 'image' in data:
                data_new = localize_df(data, dname, nproc=16)
                data_all.append(data_new)
            else:
                data_all.append(data)

        data = pd.concat(data_all)
        data['original_index'] = data.pop('index')
        data['index'] = np.arange(len(data))
        self.data = data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        idx = line['original_index']
        dname = line['SUB_DATASET']
        org_data = self.dataset_map[dname].data
        org_line = copy.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line)

    def dump_image(self, line):
        # Assert all images are pre-dumped
        assert 'image' not in line
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
        return tgt_path

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SETS)

    def evaluate(self, eval_file, **judge_kwargs):
        # First, split the eval_file by dataset
        data_all = load(eval_file)
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            data_sub = data_all[data_all['SUB_DATASET'] == dname]
            data_sub.pop('index')
            data_sub['index'] = data_sub.pop('original_index')
            data_sub.pop('SUB_DATASET')
            dump(data_sub, tgt)
        # Then, evaluate each dataset separately
        df_all = []
        dict_all = {}
        # One of the vars will be used to aggregate results
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            res = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)
            if isinstance(res, pd.DataFrame):
                res['DATASET'] = [dname] * len(res)
                df_all.append(res)
            elif isinstance(res, dict):
                res = {f'{dname}:{k}': v for k, v in res.items()}
                dict_all.update(res)
            else:
                raise NotImplementedError(f'Unknown result type {type(res)}')

        if len(df_all):
            result = pd.concat(df_all)
            score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
            dump(result, score_file)
            return result
        else:
            score_file = get_intermediate_file_path(eval_file, '_score', 'json')
            dump(dict_all, score_file)
            return dict_all


# Add new supported dataset class here
IMAGE_DATASET = [
    ImageCaptionDataset, ImageYORNDataset, ImageMCQDataset, ImageVQADataset,
    MathVision, LENS, MMMUDataset, OCRBench, MathVista, LLaVABench, LLaVABench_KO, VGRPBench, MMVet,  # noqa: E501
    MTVQADataset, TableVQABench, MMLongBench, VCRDataset, MMDUDataset, DUDE,
    SlideVQA, MUIRDataset, CCOCRDataset, GMAIMMBenchDataset, MMERealWorld,
    HRBenchDataset, CRPE, MathVerse, NaturalBenchDataset, MIABench,
    OlympiadBench, SeePhys, WildVision, MMMath, QSpatial, Dynamath, GSM8KVDataset, MMGenBench, VizWiz,  # noqa: E501
    MMNIAH, CMMMU, VLRewardBench, WeMath, LogicVista, MMMUProDataset,
    CreationMMBenchDataset, ImageShortQADataset, MMAlignBench, OmniDocBench,
    VLM2Bench, VMCBenchDataset, EMMADataset, MME_CoT, MOAT, MedXpertQA_MM_test,
    LEGO, MMSci_Captioning, Physics_yale, ScreenSpot_Pro, ScreenSpot, VenusBench_GD,
    ScreenSpotV2, OSWorld_G, VBGD, MMIFEval, Spatial457, SPARBench, SPARBenchTiny, VisuLogic, CVBench, PathVQA_VAL,
    PathVQA_TEST, TDBench, TDBenchGrounding, MicroBench, CharXiv, OmniMedVQA,
    WildDocBenchmark, MSEarthMCQ, OCR_Reasoning, PhyX, VLMBlind, CountBenchQA,
    ZEROBench, SCAM, Omni3DBench, Omni3DDetectionDataset, TallyQA, _3DSRBench, BMMR, AffordanceDataset,
    MMEReasoning, GOBenchDataset, SFE, ChartMimic, MMVMBench, XLRSBench,
    OmniEarthMCQBench, VisFactor, OSTDataset, OCRBench_v2, TreeBench, CVQA, M4Bench,
    AyaVisionBench, TopViewRS, VLMBias, MMHELIX, MedqbenchMCQDataset, MathCanvas, MMReason,
    MedqbenchPairedDescriptionDataset, MedqbenchCaptionDataset, ChartMuseum, ChartQAPro, ReasonMap_Plus,  # noqa: E501
    olmOCRBench, OceanOCRBench, MATBench, VLRMBench, RefCOCODataset, RefSpatialDataset,
    ERQADataset, SimpleVQA, HiPhODataset, MaCBench,
    UniSVG, SArena, VLMsAreBiased, MMESCIDataset, CoreCognition, GroundingME,
    FoxBench, VTCBench, Asclepius, PlotQA, ChartX, ChartBench, ChartCapDataset, WorldVQA, PuzzleVQA, VisualPuzzles,  # noqa: E501
    MMSafetyBenchDataset, MSSBenchDataset, SIUODataset, SIUOGenDataset, SIUOMCQDataset, M3oralBenchDataset,  # noqa: E501
    Design2Code, VLADBench, SSIBenchDataset, NPMM, SGI_Bench_Experimental_Reasoning, MMOral_OPG_OPEN, MMOral_OPG_CLOSED,  # noqa: E501
    CVBench, BlinkDepth, BlinkSpatial, CosmosERQA, RoboSpatialHome, SATBench, Where2Place,
    ShareRobotTrajectory, CameraIntrinsicBench, ThreeDAVGroundingBench, WarehouseSpatialAI,
    Metropolis2DDetectionDataset, Metropolis2DGroundingDataset, Astro2DDetectionDataset, HealthSurgiBench,
    LocateAnythingBench, CosmosCABImage, VANTAGE_2DPointing,
    AVSpecialOODReasoningBench, ODinW13Dataset, WorldBench,
]

# add by EASI team
IMAGE_DATASET += [
    MindCubeBench, EmbSpatialBench, ViewSpatialBench, MMSIBench, SiteBenchImage,
    SparBench, SpatialVizBench, StareBench, OmniSpatialBench, ERQABench, RoboSpatialBench, RefSpatialBench,  # noqa: E501
    SPBench, ERIQBench, DA2K
]

VIDEO_DATASET = [
    MMBenchVideo, VideoMME, MVBench, MVBench_MP4, MVTamperBench,
    LongVideoBench, WorldSense, VDC, MovieChat1k, MEGABench,
    MLVU, MLVU_MCQ, MLVU_OpenEnded,
    TempCompass, TempCompass_MCQ, TempCompass_Captioning, TempCompass_YorN,
    CGBench_MCQ_Grounding_Mini, CGBench_OpenEnded_Mini, CGBench_MCQ_Grounding, CGBench_OpenEnded,
    QBench_Video, QBench_Video_MCQ, QBench_Video_VQA,
    Video_MMLU_CAP, Video_MMLU_QA,
    Video_Holmes, VCRBench, CGAVCounting, MotionBench,
    EgoExoBench_MCQ, DREAM, VideoTT, VideoMMMU, MVUEval, OMTGBench, V2PBench, AVSpeakerBench,
    TailgatingVerification, VideoPhy2, CosmosReason, MetropolisTemporal, MetropolisVQA, MetropolisDVC,
    CausalVQA, MVPBench, IntPhys2, ITSCollision, MetropolisEventVerification, LVEventVerification, WarehouseNearMiss,
    VANTAGE_SOT, LingoQA, SOPActionRecognition, SOPTemporalLocalization,
    LVSDataset, LVSHallucinationDataset, LVSAIHallucinationDataset,
    VSSoobeClips, AVSpecialCollisionBench, AVSpecialEnvironmentBench, AVSpecialStopBehaviorBench, TemporalLocalization,
    AVPromptFollowingBench,
    AETCBench, CameraBench, CosmosCABVideoGeneral, CosmosCABVideoCamera,
]

# add by EASI team
VIDEO_DATASET += [SiteBenchVideo, VsiBench, VsiSuperRecall, VsiSuperCount, MMSIVideoBench, STIBench, DSRBench]  # noqa: E501

TEXT_DATASET = [
    TextMCQDataset, IFBench, SGI_Bench_Wet_Experiment, SGI_Bench_Dry_Experiment,
    SGI_Bench_Deep_Research, SGI_Bench_Idea_Generation, XSTestDataset, FlamesDataset
]

CUSTOM_DATASET = [
    CustomMCQDataset, CustomVQADataset, CustomTextMCQDataset
]

# Astro2DBenchDataset is a ConcatDataset subclass, so it must be built after ConcatDataset.
Astro2DBenchDataset = _build_astro2dbench_dataset(ConcatDataset)

DATASET_COLLECTION = [ConcatDataset, ConcatVideoDataset, Astro2DBenchDataset]

DATASET_CLASSES = IMAGE_DATASET + VIDEO_DATASET + TEXT_DATASET + CUSTOM_DATASET + DATASET_COLLECTION  # noqa: E501
SUPPORTED_DATASETS = []
for DATASET_CLS in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(DATASET_CLS.supported_datasets())


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'TYPE'):
                return cls.TYPE
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        TYPES = [DATASET_TYPE(dname) for dname in dataset_list]
        assert np.all([x == TYPES[0] for x in TYPES]), (dataset_list, TYPES)
        return TYPES[0]

    if 'openended' in dataset.lower():
        return 'VQA'
    warnings.warn(f'Dataset {dataset} is a custom one and not annotated as `openended`, will treat as {default}. ')  # noqa: E501
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    if dataset is None:
        warnings.warn(f'Dataset is not specified, will treat modality as {default}. ')
        return default
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'MODALITY'):
                return cls.MODALITY
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        MODALITIES = [DATASET_MODALITY(dname) for dname in dataset_list]
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (dataset_list, MODALITIES)
        return MODALITIES[0]

    if 'VIDEO' in dataset.lower():
        return 'VIDEO'
    elif 'IMAGE' in dataset.lower():
        return 'IMAGE'
    warnings.warn(f'Dataset {dataset} is a custom one, will treat modality as {default}. ')
    return default


def build_dataset(dataset_name, **kwargs):
    for cls in DATASET_CLASSES:
        if dataset_name in supported_video_datasets:
            return supported_video_datasets[dataset_name](**kwargs)
        elif dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')
    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if 'question' not in [x.lower() for x in data.columns]:
        warnings.warn(
            f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
        return None

    if 'A' in data and 'B' in data:
        if 'image' in data or 'image_path' in data:
            warnings.warn(
                f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
            return CustomMCQDataset(dataset=dataset_name, **kwargs)
        else:
            warnings.warn(
                f'Will assume unsupported dataset {dataset_name} as a Custom Text MCQ dataset. ')
            return CustomTextMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        return CustomVQADataset(dataset=dataset_name, **kwargs)


def infer_dataset_basename(dataset_name):
    basename = "_".join(dataset_name.split("_")[:-1])
    return basename


__all__ = [
    'build_dataset',
    'img_root_map',
    'build_judge',
    'extract_answer_from_item',
    'prefetch_answer',
    'DEBUG_MESSAGE',
]
__all__.extend([cls.__name__ for cls in DATASET_CLASSES])
