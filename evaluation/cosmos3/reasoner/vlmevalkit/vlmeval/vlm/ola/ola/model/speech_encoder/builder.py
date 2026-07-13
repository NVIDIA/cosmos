from .speech_encoder import DualWrappedEncoder, WhisperWrappedEncoder


def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)
    elif "dual" in speech_encoder_type.lower():
        return DualWrappedEncoder(config)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')
