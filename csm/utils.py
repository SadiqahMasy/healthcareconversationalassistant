
import torch
import torchaudio
from csm.generator import Segment
import io

def load_prompt_audio(audio_bytes: bytes, target_sample_rate: int) -> torch.Tensor:
    # Create a BytesIO object from the bytes
    audio_buffer = io.BytesIO(audio_bytes)
    
    # Load audio from the buffer instead of a file path
    audio_tensor, sample_rate = torchaudio.load(audio_buffer)
    audio_tensor = audio_tensor.squeeze(0)
    
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: bytes, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)