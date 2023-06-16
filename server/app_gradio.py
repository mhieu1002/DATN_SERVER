import os
import logging
from logging.config import dictConfig
import torch
import torchaudio
from io import BytesIO
import numpy as np
from inference import parse_arguments, load_models, infer_mb_melgan, split_sentences
from common.text.text_processing import TextProcessing
from pydub import AudioSegment
import scipy.io.wavfile as write_wav
from denoiser import pretrained
from denoiser.dsp import convert_audio
from IPython import display as disp
import re
import pydub
from pydub.silence import split_on_silence
from scipy.signal import butter, filtfilt
from underthesea import sent_tokenize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arguments = [
    '-i',
    'vi_phrases/devset.tsv',
    '-o',
    './vi_output/audio',
    '--fastpitch',
    'checkpoints/FastPitch_checkpoint_40.pt',
    '--mbmelgan',
    'checkpoints/UTC2TTS_13efcb4_0575.pt',
    '--waveglow',
    'SKIP',
    '--batch-size',
    '16',
    '--denoising-strength',
    '0.01',
    '--repeats',
    '1',
    '--warmup-steps',
    '0',
    '--speaker',
    '0',
    '--n-speakers',
    '1',
    '--energy-conditioning']

parser, args, unk_args = parse_arguments(arguments)
tp = TextProcessing(args.symbol_set, args.text_cleaners)
load_models(parser, arguments, args, unk_args)
remove_noise_model = pretrained.dns64()

MAX_LENGTH = 30

def text_to_speech(text_input):
    text_cleaned = tp.clean_text(text_input)
    if text_cleaned[-1] not in '?.!':
        text_cleaned += '.'
    # sentences = sent_tokenize(text_cleaned)
    text_cleaned = re.sub("(?<![.!?])[\n]+", ".\n", text_cleaned)
    sentences, breaks = split_sentences(text_cleaned, MAX_LENGTH)

    audios = []
    
    for sentence in sentences:
        result = infer_mb_melgan([sentence], args)
        audio, rate, perf = result
        audio = audio[0]
        audios.append(audio)
        
    audio_combined = np.concatenate(audios)
    audio_combined = audio_combined.astype(np.int16)
    input_path = 'output/test.wav'
    write_wav.write(input_path, rate, audio_combined)
    
    # Load and process the audio file
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(22050)  # Set sample rate to 22050 Hz
    
    # Split audio on silence
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-36, keep_silence=500)
    
    # Process and filter chunks
    processed_chunks = []
    for chunk in chunks:
        # Apply additional noise reduction or filtering if needed
        processed_chunk = chunk
        processed_chunk = processed_chunk.low_pass_filter(1000)  # Apply a low-pass filter
        processed_chunk = processed_chunk.normalize()  # Normalize the audio volume
        processed_chunks.append(processed_chunk)
    
    # Concatenate processed chunks
    combined_audio = processed_chunks[0]
    for segment in processed_chunks[1:]:
        combined_audio += segment
    
    # Export the final audio
    save_tmp = 'output/test1.wav'
    combined_audio.export(save_tmp, format="wav")
    
    return save_tmp

def process(txt_test):
    return text_to_speech(txt_test)