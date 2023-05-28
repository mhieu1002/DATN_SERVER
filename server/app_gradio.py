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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arguments = [
    '-i',
    'vi_phrases/devset.tsv',
    '-o',
    './vi_output/audio',
    '--fastpitch',
    'checkpoints/FastPitch_checkpoint_40.pt',
    '--mbmelgan',
    'checkpoints/UTC2TTS_13efcb4_0375.pt',
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
    text_cleaned = re.sub("(?<![.!?])[\n]+", ".\n", text_cleaned)
    sentences, breaks = split_sentences(text_cleaned, MAX_LENGTH)
    result = infer_mb_melgan([text_cleaned], args)
    audios, rate, perf = result
    audios = list(audios)
    # for i, _break in enumerate(breaks):
        # audios.insert(_break+i, list(AudioSegment.silent(500, 22050).raw_data))
        
    audio = np.concatenate(audios)
    #audio /= (1.414 / 32767)
    audio = audio.astype(np.int16)
    input_path='output/test.wav'
    write_wav.write(input_path, rate, audio)
    
    wav, sr = torchaudio.load(input_path)
    wav_save = convert_audio(wav, sr, 22050, remove_noise_model.chin)
    with torch.no_grad():
        denoised = remove_noise_model(wav_save[None])[0]
    audio_final = disp.Audio(denoised.data.cpu().numpy(), rate=22050)
    save_tmp='output/test1.wav'
    with open(save_tmp, 'wb') as fa:
        fa.write(audio_final.data)
    
    return save_tmp

def process(txt_test):
    return text_to_speech(txt_test)