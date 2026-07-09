from functools import partial

from vlmeval.dataset import *

vcrbench_dataset = {
    'VCRBench_8frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=8, pack=False),
    'VCRBench_16frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=16, pack=False),
    'VCRBench_32frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=32, pack=False),
    'VCRBench_64frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=64, pack=False),
    'VCRBench_1fps_nopack': partial(VCRBench, dataset='VCR-Bench', fps=1.0, pack=False)
}

v2pbench_dataset = {
    'V2PBench_2frame_nopack': partial(V2PBench, dataset='V2P-Bench', nframe=2, pack=False),
    'V2PBench_8frame_nopack': partial(V2PBench, dataset='V2P-Bench', nframe=8, pack=False),
    'V2PBench_16frame_nopack': partial(V2PBench, dataset='V2P-Bench', nframe=16, pack=False),
    'V2PBench_64frame_nopack': partial(V2PBench, dataset='V2P-Bench', nframe=64, pack=False),
    'V2PBench_128frame_nopack': partial(V2PBench, dataset='V2P-Bench', nframe=128, pack=False),
    'V2PBench_1fps_nopack': partial(V2PBench, dataset='V2P-Bench', fps=1.0, pack=False)
}

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_64frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=True),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': partial(MVBench, dataset='MVBench', nframe=64),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

tamperbench_dataset = {
    'MVTamperBench_8frame': partial(MVTamperBench, dataset='MVTamperBench', nframe=8),
    'MVTamperBenchStart_8frame': partial(MVTamperBench, dataset='MVTamperBenchStart', nframe=8),
    'MVTamperBenchEnd_8frame': partial(MVTamperBench, dataset='MVTamperBenchEnd', nframe=8),
}

videomme_dataset = {
    'Video-MME_8frame': partial(VideoMME, dataset='Video-MME', nframe=8),
    'Video-MME_64frame': partial(VideoMME, dataset='Video-MME', nframe=64),
    'Video-MME_8frame_subs': partial(VideoMME, dataset='Video-MME', nframe=8, use_subtitle=True),
    'Video-MME_1fps': partial(VideoMME, dataset='Video-MME', fps=1.0),
    'Video-MME_0.5fps': partial(VideoMME, dataset='Video-MME', fps=0.5),
    'Video-MME_0.5fps_subs': partial(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
}

videommmu_dataset = {
    'VideoMMMU_8frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=8),
    'VideoMMMU_64frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=64),
    'VideoMMMU_1fps': partial(VideoMMMU, dataset='VideoMMMU', fps=1.0),
    'VideoMMMU_0.5fps': partial(VideoMMMU, dataset='VideoMMMU', fps=0.5),
}

longvideobench_dataset = {
    'LongVideoBench_8frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=8),
    'LongVideoBench_8frame_subs': partial(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
    'LongVideoBench_64frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=64),
    'LongVideoBench_1fps': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0),
    'LongVideoBench_0.5fps': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5),
    'LongVideoBench_0.5fps_subs': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True)
}

mlvu_dataset = {
    'MLVU_8frame': partial(MLVU, dataset='MLVU', nframe=8),
    'MLVU_64frame': partial(MLVU, dataset='MLVU', nframe=64),
    'MLVU_1fps': partial(MLVU, dataset='MLVU', fps=1.0)
}

tempcompass_dataset = {
    'TempCompass_8frame': partial(TempCompass, dataset='TempCompass', nframe=8),
    'TempCompass_64frame': partial(TempCompass, dataset='TempCompass', nframe=64),
    'TempCompass_1fps': partial(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_0.5fps': partial(TempCompass, dataset='TempCompass', fps=0.5)
}

default = {
    # "AVSpecialCollisionBench": partial(AVSpecialCollisionBench, dataset='AVSpecialCollisionBench', fps=30, total_pixels=8192 * 28 * 28),
    # "AVSpecialStopBehaviorBench": partial(AVSpecialStopBehaviorBench, dataset='AVSpecialStopBehaviorBench', fps=30, total_pixels=8192 * 28 * 28),
    "TemporalLocalization": partial(TemporalLocalization, dataset='TemporalLocalization', fps=8.0, total_pixels=8192 * 28 * 28),
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
# We implement the metrics long_acc, clue_acc, miou, CRR, acc@iou and rec@iou
# in the CGBench_MCQ_Grounding_Mini and CGBench_MCQ_Grounding datasets;
# the metric open-ended is implemented in the CGBench_OpenEnded_Mini and CGBench_OpenEnded datasets.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': partial(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': partial(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}

megabench_dataset = {
    'MEGABench_core_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="core"),
    'MEGABench_open_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="open"),
    'MEGABench_core_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="core"),
    'MEGABench_open_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="open")
}

moviechat1k_dataset = {
    'moviechat1k_breakpoint_8frame': partial(MovieChat1k, dataset='MovieChat1k', subset='breakpoint', nframe=8),
    'moviechat1k_global_14frame': partial(MovieChat1k, dataset='MovieChat1k', subset='global', nframe=14),
    'moviechat1k_global_8frame_limit0.01': partial(
        MovieChat1k, dataset='MovieChat1k', subset='global', nframe=8, limit=0.01
    )
}

vdc_dataset = {
    'VDC_8frame': partial(VDC, dataset='VDC', nframe=8),
    'VDC_1fps': partial(VDC, dataset='VDC', fps=1.0),
}

worldsense_dataset = {
    'WorldSense_8frame': partial(WorldSense, dataset='WorldSense', nframe=8),
    'WorldSense_8frame_subs': partial(WorldSense, dataset='WorldSense', nframe=8, use_subtitle=True),
    'WorldSense_8frame_audio': partial(WorldSense, dataset='WorldSense', nframe=8, use_audio=True),
    'WorldSense_32frame': partial(WorldSense, dataset='WorldSense', nframe=32),
    'WorldSense_32frame_subs': partial(WorldSense, dataset='WorldSense', nframe=32, use_subtitle=True),
    'WorldSense_32frame_audio': partial(WorldSense, dataset='WorldSense', nframe=32, use_audio=True),
    'WorldSense_1fps': partial(WorldSense, dataset='WorldSense', fps=1.0),
    'WorldSense_1fps_subs': partial(WorldSense, dataset='WorldSense', fps=1.0, use_subtitle=True),
    'WorldSense_1fps_audio': partial(WorldSense, dataset='WorldSense', fps=1.0, use_audio=True),
    'WorldSense_0.5fps': partial(WorldSense, dataset='WorldSense', fps=0.5),
    'WorldSense_0.5fps_subs': partial(WorldSense, dataset='WorldSense', fps=0.5, use_subtitle=True),
    'WorldSense_0.5fps_audio': partial(WorldSense, dataset='WorldSense', fps=0.5, use_audio=True)
}

qbench_video_dataset = {
    'QBench_Video_8frame': partial(QBench_Video, dataset='QBench_Video', nframe=8),
    'QBench_Video_16frame': partial(QBench_Video, dataset='QBench_Video', nframe=16),
}

video_mmlu_dataset = {
    'Video_MMLU_CAP_16frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=16),
    'Video_MMLU_CAP_64frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=64),
    'Video_MMLU_QA_16frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=16),
    'Video_MMLU_QA_64frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=64),
}

video_tt_dataset = {
    'Video_TT_16frame': partial(VideoTT, dataset='Video-TT', nframe=16),
    'Video_TT_32frame': partial(VideoTT, dataset='Video-TT', nframe=32),
    'Video_TT_64frame': partial(VideoTT, dataset='Video-TT', nframe=64),
}

video_holmes_dataset = {
    'Video_Holmes_32frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=32),
    'Video_Holmes_64frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=64),
}

motionbench_dataset = {
    'MotionBench_8frame':  partial(MotionBench, dataset='MotionBench', nframe=8),
    'MotionBench_16frame': partial(MotionBench, dataset='MotionBench', nframe=16),
    'MotionBench_32frame': partial(MotionBench, dataset='MotionBench', nframe=32),
    'MotionBench_64frame': partial(MotionBench, dataset='MotionBench', nframe=64),
    'MotionBench_1fps':    partial(MotionBench, dataset='MotionBench', fps=1.0),
}

cg_av_counting_dataset = {
    'CG-AV-Counting_32frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=32, use_frame_time=False),
    'CG-AV-Counting_64frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=64, use_frame_time=False)
}

egoexobench_dataset = {
    'EgoExoBench_64frame': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=False),  # noqa: E501
    'EgoExoBench_64frame_skip_EgoExo4D': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=True)  # noqa: E501

}

dream_1k_dataset = {
    'DREAM-1K_8frame': partial(DREAM, dataset='DREAM-1K', nframe=8),
    'DREAM-1K_64frame': partial(DREAM, dataset='DREAM-1K', nframe=64),
    'DREAM-1K_2fps': partial(DREAM, dataset='DREAM-1K', fps=2.0),
    'DREAM-1K_1fps': partial(DREAM, dataset='DREAM-1K', fps=1.0),
    'DREAM-1K_0.5fps': partial(DREAM, dataset='DREAM-1K', fps=0.5),
}

av_speakerbench_dataset = {
    # frame-sampled variants
    'AV-SpeakerBench_audiovisual_8frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=True
    ),
    'AV-SpeakerBench_audiovisual_16frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=True
    ),
    'AV-SpeakerBench_visual_8frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=False
    ),
    'AV-SpeakerBench_visual_16frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=False
    ),
    'AV-SpeakerBench_audio_only_8frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=True, audio_only=True
    ),
    'AV-SpeakerBench_audio_only_16frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=True, audio_only=True
    ),
    # fps-based variants
    'AV-SpeakerBench_audiovisual_1fps': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=True
    ),
    'AV-SpeakerBench_visual_1fps': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=False
    ),
    'AV-SpeakerBench_audio_only_1fps': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=True, audio_only=True
    ),
    # shorthand aliases mapping to audiovisual
    'AV-SpeakerBench_8frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=8, use_audio=True
    ),
    'AV-SpeakerBench_16frame': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', nframe=16, use_audio=True
    ),
    'AV-SpeakerBench_1fps': partial(
        AVSpeakerBench, dataset='AV-SpeakerBench', fps=1.0, use_audio=True
    ),
}

omtg_dataset = {
    "OMTGBench_1fps": partial(OMTGBench, dataset="OMTGBench", fps=1.0),
    "OMTGBench_2fps": partial(OMTGBench, dataset="OMTGBench", fps=2.0),
}

mvu_eval_dataset = {
    'MVU-Eval_8frame': partial(MVUEval, dataset='MVU-Eval', nframe=8),
    'MVU-Eval_16frame': partial(MVUEval, dataset='MVU-Eval', nframe=16),
}

VSI_FRAME_VARIANTS = [
    ("128frame", dict(nframe=128)),
    ("64frame", dict(nframe=64)),
    ("32frame", dict(nframe=32)),
    ("16frame", dict(nframe=16)),
    ("2fps", dict(fps=2.0)),
    ("1fps", dict(fps=1.0)),
]


def _build_video_variants(subsets, cls, variants=VSI_FRAME_VARIANTS):
    out = {}
    for variant in subsets:
        for suffix, params in variants:
            out[f"{variant}_{suffix}"] = partial(cls, dataset=variant, **params)
    return out


# === VSI-Bench ===
vsi_subsets = VsiBench.supported_datasets()
video_vsi_dataset = _build_video_variants(vsi_subsets, VsiBench)

# === VSI-SUPER-Recall ===
vsisuper_recall_subsets = VsiSuperRecall.supported_datasets()
vsisuper_recall_dataset = _build_video_variants(vsisuper_recall_subsets, VsiSuperRecall)

# === VSI-SUPER-Count ===
vsisuper_count_subsets = VsiSuperCount.supported_datasets()
vsisuper_count_dataset = _build_video_variants(vsisuper_count_subsets, VsiSuperCount)

sitebenchvideo_dataset = {
    'SiteBenchVideo_64frame': partial(SiteBenchVideo, dataset='SiteBenchVideo', nframe=64),
    'SiteBenchVideo_32frame': partial(SiteBenchVideo, dataset='SiteBenchVideo', nframe=32),
    'SiteBenchVideo_1fps': partial(SiteBenchVideo, dataset='SiteBenchVideo', fps=1),
}

mmsi_video_dataset = {
    # The 300 frame setting is aligned with Sufficient-Coverage policy proposed in MMSI-Video-Bench paper
    'MMSIVideoBench_300frame': partial(MMSIVideoBench, dataset='MMSIVideoBench', nframe=300),
    'MMSIVideoBench_64frame': partial(MMSIVideoBench, dataset='MMSIVideoBench', nframe=64),
    'MMSIVideoBench_50frame': partial(MMSIVideoBench, dataset='MMSIVideoBench', nframe=50),
    'MMSIVideoBench_32frame': partial(MMSIVideoBench, dataset='MMSIVideoBench', nframe=32),
    'MMSIVideoBench_1fps': partial(MMSIVideoBench, dataset='MMSIVideoBench', fps=1),
}

sti_subsets = STIBench.supported_datasets()
sti_variants = [
    ("64frame", dict(nframe=64)),
    ("32frame", dict(nframe=32)),
    # The 30 frame setting is aligned with offical seting STI-Bench paper
    ("30frame", dict(nframe=30)),
    ("1fps", dict(fps=1.0)),
]
sti_dataset = _build_video_variants(sti_subsets, STIBench, sti_variants)

dsr_subsets = DSRBench.supported_datasets()
dsr_variants = [
    ("64frame", dict(nframe=64)),
    ("32frame", dict(nframe=32)),
    ("30frame", dict(nframe=30)),
    # The 1fps setting is aligned with offical seting DSR-Bench paper
    ("1fps", dict(fps=1.0)),
]
dsr_dataset = _build_video_variants(dsr_subsets, DSRBench, dsr_variants)

# Fork-specific video dataset configs
videophy2_dataset = {
    # Use FPS and total_pixels in cosmos_reason1 example
    'VideoPhy2': partial(VideoPhy2, dataset='VideoPhy2', fps=16.0, total_pixels=8192 * 28 * 28),
}

metropolis_temporal_dataset = {
    # All categories configuration (no filtering - evaluates all samples in TSV)
    'MetropolisTemporal_all': partial(MetropolisTemporal, dataset='MetropolisTemporal', fps=1.0),

    # Default configuration (matching reference: nframe=0, fps=1)
    'MetropolisTemporal': partial(MetropolisTemporal, dataset='MetropolisTemporal', fps=1.0, max_frames=256, total_pixels=16384 * 32 * 32, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_4fps': partial(MetropolisTemporal, dataset='MetropolisTemporal', fps=4.0, max_frames=256, total_pixels=16384 * 32 * 32, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_10fps': partial(MetropolisTemporal, dataset='MetropolisTemporal', fps=10.0, max_frames=256, total_pixels=16384 * 32 * 32, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),

    # Frame-based configurations (for comparison, but not in reference)
    'MetropolisTemporal_8frame': partial(MetropolisTemporal, dataset='MetropolisTemporal', nframe=8, fps=-1, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_16frame': partial(MetropolisTemporal, dataset='MetropolisTemporal', nframe=16, fps=-1, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_32frame': partial(MetropolisTemporal, dataset='MetropolisTemporal', nframe=32, fps=-1, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_64frame': partial(MetropolisTemporal, dataset='MetropolisTemporal', nframe=64, fps=-1, total_pixels=8192 * 32 * 32, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),

    # Test configurations with limited samples
    'MetropolisTemporal_test3_fps': partial(MetropolisTemporal, dataset='MetropolisTemporal', limit=3, fps=1.0, nframe=0, verbose=True, random_state=42, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_test10': partial(MetropolisTemporal, dataset='MetropolisTemporal', limit=5, verbose=True, total_pixels=8192 * 32 * 32, nframe=64, random_state=42, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_test50': partial(MetropolisTemporal, dataset='MetropolisTemporal', limit=50, nframe=64, fps=0, random_state=42, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_test100': partial(MetropolisTemporal, dataset='MetropolisTemporal', limit=100, nframe=64, fps=0, random_state=42, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisTemporal_test1p': partial(MetropolisTemporal, dataset='MetropolisTemporal', limit=0.01, nframe=64, fps=0, random_state=42, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),  # 1% of data

    # Verbose test configurations for debugging
    'MetropolisTemporal_test10_verbose': partial(
        MetropolisTemporal,
        dataset='MetropolisTemporal',
        limit=10,
        verbose=True,
        total_pixels=8192 * 32 * 32,
        nframe=64,
        random_state=42,
        include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']
    ),
    'MetropolisTemporal_test200_verbose': partial(
        MetropolisTemporal,
        dataset='MetropolisTemporal',
        limit=200,
        verbose=True,
        fps=1.0,
        random_state=42,
        include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']
    ),
}

causalvqa_dataset = {
    'CausalVQA': partial(CausalVQA, dataset='CausalVQA', nframe=0, fps=1.0, total_pixels=8192 * 32 * 32, max_frames=256),
    'CausalVQA_4fps': partial(CausalVQA, dataset='CausalVQA', nframe=0, fps=4.0, total_pixels=8192 * 32 * 32, max_frames=256),
    'CausalVQA_8fps': partial(CausalVQA, dataset='CausalVQA', nframe=0, fps=8.0, total_pixels=16384 * 32 * 32, max_frames=256),
}

mvpbench_dataset = {'MVPBench': partial(MVPBench, dataset='MVPBench', fps=4.0, total_pixels=8192 * 32 * 32)}

metropolis_dvc_dataset = {
    # Default configuration - downloads from DSS on first run
    'MetropolisDVC': partial(MetropolisDVC, dataset='MetropolisDVC', fps=4.0, max_frames=128, total_pixels=8192 * 32 * 32),

    # FPS-based configurations
    'MetropolisDVC_1fps': partial(MetropolisDVC, dataset='MetropolisDVC', fps=1.0, max_frames=128, total_pixels=8192 * 32 * 32),
    'MetropolisDVC_2fps': partial(MetropolisDVC, dataset='MetropolisDVC', fps=2.0, max_frames=128, total_pixels=8192 * 32 * 32),
    'MetropolisDVC_4fps': partial(MetropolisDVC, dataset='MetropolisDVC', fps=4.0, max_frames=128, total_pixels=8192 * 32 * 32),

    # Frame-based configurations
    'MetropolisDVC_8frame': partial(MetropolisDVC, dataset='MetropolisDVC', nframe=8, fps=-1),
    'MetropolisDVC_64frame': partial(MetropolisDVC, dataset='MetropolisDVC', nframe=64, fps=-1),

}

metropolis_vqa_dataset = {
    'MetropolisVQA': partial(MetropolisVQA, dataset='MetropolisVQA', fps=1.0),
    'MetropolisVQA_realworld': partial(MetropolisVQA, dataset='MetropolisVQA', fps=1.0, max_frames=256, total_pixels=16384 * 32 * 32, include_categories=['Transportation_real', 'Transportation', 'Smart_Spaces', 'Warehouse']),
    # 'MetropolisVQA_realworld': partial(MetropolisVQA, dataset='MetropolisVQA', fps=1.0, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    # 'MetropolisVQA_realworld': partial(MetropolisVQA, dataset='MetropolisVQA', fps=1.0, max_frames=256, total_pixels=16384 * 32 * 32, include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse']),
    'MetropolisVQA_transportation': partial(MetropolisVQA, dataset='MetropolisVQA', fps=1.0, include_categories=['Transportation_real', 'Transportation']),
}


# VANTAGE-bench variants. Each `dataset='VANTAGE_*'` arg triggers the
# per-dataset-name _S3_PATHS dispatch inside the refactored Metropolis* classes,
# routing the fetch to the HF-release-with-annotations stage on team-cosmos S3.
# Sampling values mirror the upstream VANTAGE-source partials verbatim. These
# affect vlmevalkit-direct callers (`run.py --data VANTAGE_VQA_8frame`); they
# don't apply on the vlmeval-metric path, which dispatches via configs/*.json.
vantage_vqa_dataset = {
    'VANTAGE_VQA_8frame':     partial(MetropolisVQA, dataset='VANTAGE_VQA', nframe=8),
    'VANTAGE_VQA_16frame':    partial(MetropolisVQA, dataset='VANTAGE_VQA', nframe=16),
    'VANTAGE_VQA_64frame':    partial(MetropolisVQA, dataset='VANTAGE_VQA', nframe=64),
    'VANTAGE_VQA_1fps':       partial(MetropolisVQA, dataset='VANTAGE_VQA', fps=1.0),
    'VANTAGE_VQA_0.5fps':     partial(MetropolisVQA, dataset='VANTAGE_VQA', fps=0.5),
    'VANTAGE_VQA_8frame_200': partial(MetropolisVQA, dataset='VANTAGE_VQA', nframe=8, limit=200, random_state=42),
}

vantage_temporal_dataset = {
    'VANTAGE_Temporal_8frame':  partial(MetropolisTemporal, dataset='VANTAGE_Temporal', nframe=8,  total_pixels=8192 * 32 * 32,  max_frames=256),
    'VANTAGE_Temporal_16frame': partial(MetropolisTemporal, dataset='VANTAGE_Temporal', nframe=16, total_pixels=8192 * 32 * 32,  max_frames=256),
    'VANTAGE_Temporal_64frame': partial(MetropolisTemporal, dataset='VANTAGE_Temporal', nframe=64, total_pixels=8192 * 32 * 32,  max_frames=256),
    'VANTAGE_Temporal_1fps':    partial(MetropolisTemporal, dataset='VANTAGE_Temporal', fps=1.0,   total_pixels=16384 * 32 * 32, max_frames=256),
    'VANTAGE_Temporal_0.5fps':  partial(MetropolisTemporal, dataset='VANTAGE_Temporal', fps=0.5,   total_pixels=16384 * 32 * 32, max_frames=256),
}

vantage_dvc_dataset = {
    'VANTAGE_DVC_8frame':  partial(MetropolisDVC, dataset='VANTAGE_DVC', nframe=8,  total_pixels=8192 * 32 * 32, max_frames=128),
    'VANTAGE_DVC_64frame': partial(MetropolisDVC, dataset='VANTAGE_DVC', nframe=64, total_pixels=8192 * 32 * 32, max_frames=128),
    'VANTAGE_DVC_1fps':    partial(MetropolisDVC, dataset='VANTAGE_DVC', fps=1.0,   total_pixels=8192 * 32 * 32, max_frames=128),
    'VANTAGE_DVC_2fps':    partial(MetropolisDVC, dataset='VANTAGE_DVC', fps=2.0,   total_pixels=8192 * 32 * 32, max_frames=128),
    'VANTAGE_DVC_4fps':    partial(MetropolisDVC, dataset='VANTAGE_DVC', fps=4.0,   total_pixels=8192 * 32 * 32, max_frames=128),
}

vantage_event_verification_dataset = {
    'VANTAGE_EventVerification_8frame':  partial(MetropolisEventVerification, dataset='VANTAGE_EventVerification', nframe=8,  fps=0,    total_pixels=8192 * 32 * 32,  max_frames=256),
    'VANTAGE_EventVerification_16frame': partial(MetropolisEventVerification, dataset='VANTAGE_EventVerification', nframe=16, fps=0,    total_pixels=8192 * 32 * 32,  max_frames=256),
    'VANTAGE_EventVerification_1fps':    partial(MetropolisEventVerification, dataset='VANTAGE_EventVerification', fps=1.0,             total_pixels=16384 * 32 * 32, max_frames=256),
}


lingoqa_dataset = {
    'LingoQA': partial(LingoQA, dataset='LingoQA', nframe=6),
}


lvs_dataset = {
    # LVS dataset configurations using Metropolis LVS data format
    # Structure: case_id/contextual/video_id/events.json + case_id/raw/video_id.mp4
    # LVS.tsv is auto-generated from events.json and video.json if not present
    'LVS': partial(LVSDataset, fps=-1, nframe=500, nvdataset_name='external-chicago-copa-body-worn-camera'),
    'LVS_500frames': partial(LVSDataset, fps=-1, nframe=500, nvdataset_name='external-chicago-copa-body-worn-camera'),
    'LVS_1fps': partial(LVSDataset, fps=1.0, nvdataset_name='external-chicago-copa-body-worn-camera'),
}

lvs_hallucination_dataset = {
    # Default: 10-second chunks, 1 fps (10 frames per chunk)
    'LVS_hallucination': partial(
        LVSHallucinationDataset,
        nvdataset_name='external-chicago-copa-body-worn-camera',
        chunk_duration=10,
        fps=1.0,
    ),
    'LVS_hallucination20fpc': partial(
        LVSHallucinationDataset,
        dataset='LVS_hallucination20fpc',
        nvdataset_name='external-chicago-copa-body-worn-camera',
        chunk_duration=10,
        fps=2.0,
        scenario="Police body-worn camera footage",
    ),
    'LVS_hallucination40fpc': partial(
        LVSHallucinationDataset,
        dataset='LVS_hallucination40fpc',
        nvdataset_name='external-chicago-copa-body-worn-camera',
        chunk_duration=10,
        fps=4.0,
        scenario="Police body-worn camera footage",
    ),
    # 10-second chunks with 5 frames per chunk
    'LVS_hallucination_10s_5frame': partial(
        LVSHallucinationDataset,
        nvdataset_name='external-chicago-copa-body-worn-camera',
        chunk_duration=10,
        fps=-1,
        nframe=5,
    ),
    # 30-second chunks with 1 fps (30 frames per chunk)
    'LVS_hallucination_30s_1fps': partial(
        LVSHallucinationDataset,
        nvdataset_name='external-chicago-copa-body-worn-camera',
        chunk_duration=30,
        fps=1.0,
    ),
    # 10-second chunks with 2 fps (20 frames per chunk)
    'LVS_hallucination_10s_2fps': partial(
        LVSHallucinationDataset,
        nvdataset_name='external-chicago-copa-body-worn-camera',
        chunk_duration=10,
        fps=2.0,
    ),
}

# LVS AI Hallucination dataset (LLM-judge based evaluation)
lvs_ai_hallucination_dataset = {
    'LVS_ai_hallucination': partial(
        LVSAIHallucinationDataset,
        nvdataset_name='external-chicago-copa-body-worn-camera',
        chunk_duration=10,
        fps=2.0,
        scenario="Police body-worn camera footage",
        eval_model="qwen3_235b_a22b",
        frames_per_chunk=20,
    ),
}

tailgating_verification_dataset = {
    'tailgating_courtyard': partial(TailgatingVerification, dataset='tailgating_courtyard', fps=4, max_frames=128, max_pixels=186625, system_prompt_option="merged"),
    'tailgating_building_r': partial(TailgatingVerification, dataset='tailgating_building_r', fps=4, max_frames=128, max_pixels=186625, system_prompt_option="merged"),
}

camera_bench_dataset = {
    # CameraBench: 3 API calls per video (1 batch binary + 2 multiclass).
    # CameraBenchDesc: 1 API call per video (open-ended description, parse all tags).
    'CameraBench':     partial(CameraBench, dataset='CameraBench',     fps=4.0, variant='B'),
    'CameraBenchDesc': partial(CameraBench, dataset='CameraBenchDesc', fps=4.0, variant='desc'),
}

cosmos_cab_video_dataset = {
    # Cosmos-CAB-Video_General: 341 clips. precision (decompose + multimodal verify) + recall (assertion check).
    # Cosmos-CAB-Video_Camera:  160 clips. caption -> tag prediction -> macro F1 vs GT.
    'Cosmos-CAB-Video_General': partial(CosmosCABVideoGeneral, dataset='Cosmos-CAB-Video_General', fps=4.0),
    'Cosmos-CAB-Video_Camera':  partial(CosmosCABVideoCamera,  dataset='Cosmos-CAB-Video_Camera',  fps=4.0),
}

aetcbench_dataset = {
    'AETCBench_all': partial(AETCBench, dataset='AETCBench', split='test', task='all', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_bcq': partial(AETCBench, dataset='AETCBench', split='test', task='bcq', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_mcq': partial(AETCBench, dataset='AETCBench', split='test', task='mcq', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_open_qa': partial(AETCBench, dataset='AETCBench', split='test', task='open_qa', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_temporal_localization': partial(AETCBench, dataset='AETCBench', split='test', task='temporal_localization', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_causal_linkage': partial(AETCBench, dataset='AETCBench', split='test', task='causal_linkage', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_scene_description': partial(AETCBench, dataset='AETCBench', split='test', task='scene_description', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_temporal_description': partial(AETCBench, dataset='AETCBench', split='test', task='temporal_description', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_video_summarization': partial(AETCBench, dataset='AETCBench', split='test', task='video_summarization', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_bcq_openended': partial(AETCBench, dataset='AETCBench', split='test', task='bcq_openended', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
    # 'AETCBench_mcq_openended': partial(AETCBench, dataset='AETCBench', split='test', task='mcq_openended', fps=4, preprocess_fps=4, preprocess_max_pixels=720 * 1280),
}

supported_video_datasets = {}

dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, videommmu_dataset, longvideobench_dataset,
    mlvu_dataset, tempcompass_dataset, cgbench_dataset, worldsense_dataset, tamperbench_dataset,
    megabench_dataset, qbench_video_dataset, moviechat1k_dataset, vdc_dataset, video_holmes_dataset, vcrbench_dataset,
    motionbench_dataset,
    cg_av_counting_dataset, video_mmlu_dataset, egoexobench_dataset, dream_1k_dataset, video_tt_dataset,
    video_vsi_dataset, mvu_eval_dataset, omtg_dataset, v2pbench_dataset, av_speakerbench_dataset,
    videophy2_dataset, metropolis_temporal_dataset,
    metropolis_vqa_dataset, metropolis_dvc_dataset,
    vantage_vqa_dataset, vantage_temporal_dataset, vantage_dvc_dataset, vantage_event_verification_dataset,
    causalvqa_dataset, mvpbench_dataset, lingoqa_dataset,
    lvs_dataset, lvs_hallucination_dataset, lvs_ai_hallucination_dataset,
    tailgating_verification_dataset, aetcbench_dataset, camera_bench_dataset, cosmos_cab_video_dataset, default
]

# add by EASI team
dataset_groups += [
    sitebenchvideo_dataset, mmsi_video_dataset, vsisuper_recall_dataset, vsisuper_count_dataset,
    sti_dataset, dsr_dataset
]

for grp in dataset_groups:
    supported_video_datasets.update(grp)
