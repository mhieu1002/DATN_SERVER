import argparse
import re
from typing import Any
import models
import time
import os
import sys
import warnings
from pathlib import Path
import torch
import numpy as np
import json
import string
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from common.text import cmudict
from common.text.text_processing import TextProcessing
from waveglow.denoiser import Denoiser
#from hifigan.models import Generator
import waveglow.waveglow as glow
from mbmelgan.models.generator import Generator
sys.modules['glow'] = glow
import logging
logger = logging.getLogger("myapp")
denoiser: Any
waveglow: Any
hifigan: Any
generator: Any
device_used: Any

MAX_WAV_VALUE = 32768.0


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=False,
                        help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true', help='')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str,
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('--waveglow', type=str,
                        help='Full path to the WaveGlow model checkpoint file (skip to only generate mels)')
    parser.add_argument('--hifigan', type=str,
                        help='Full path to the HifiGan model checkpoint file (skip to only generate mels)')
    parser.add_argument('--mbmelgan', type=str,
                        help='Full path to the Multi-band MelGAN model checkpoint file (skip to only generate mels)')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
                        help='WaveGlow denoising')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    parser.add_argument('--p-arpabet', type=float, default=0.0, help='')
    parser.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                        help='')
    parser.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
                        help='')
    transform = parser.add_argument_group('transform')
    transform.add_argument('--fade-out', type=int, default=10,
                           help='Number of fadeout frames at the end')
    transform.add_argument('--pace', type=float, default=1.0,
                           help='Adjust the pace of speech')
    transform.add_argument('--pitch-transform-flatten', action='store_true',
                           help='Flatten the pitch')
    transform.add_argument('--pitch-transform-invert', action='store_true',
                           help='Invert the pitch wrt mean value')
    transform.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                           help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
    transform.add_argument('--pitch-transform-shift', type=float, default=0.0,
                           help='Raise/lower the pitch by <hz>')
    transform.add_argument('--pitch-transform-custom', action='store_true',
                           help='Apply the transform from pitch_transform.py')

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['vietnamese_cleaners_v2'], type=str,
                                 help='Type of text cleaners for input text')
    text_processing.add_argument('--symbol-set', type=str, default='vietnamese_basic',
                                 help='Define symbol set for input text')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the model.')

    return parser



class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(3)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(3)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)

#Load Hparam
# modified from https://github.com/HarryVolek/PyTorch_Speaker_Verification

import yaml

def load_hparam_str(hp_str):
    path = 'temp-restore.yaml'
    with open(path, 'w') as f:
        f.write(hp_str)
    ret = HParam(path)
    os.remove(path)
    return ret


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class HParam(Dotdict):

    def __init__(self, file):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Pseudo QMF modules."""

import numpy as np
import torch
import torch.nn.functional as F

from scipy.signal import kaiser


def design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) \
            / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h

class PQMF(torch.nn.Module):
    """PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
        """Initilize PQMF module.
        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.
        """
        super(PQMF, self).__init__()

        # define filter coefficient
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - ((taps - 1) / 2)) +
                (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - ((taps - 1) / 2)) -
                (-1) ** k * np.pi / 4)

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)


def load_model_from_ckpt(checkpoint_path, device, ema, model):
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')

    return model


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_mb_melgan(checkpoint_path):
    global hp
    #load multi-band melGAN
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hp = load_hparam_str(checkpoint['hp_str'])

    vocoder = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                            ratios=hp.model.generator_ratio, mult = hp.model.mult,
                            out_band = hp.model.out_channels)
    vocoder.load_state_dict(checkpoint['model_g'])
    vocoder.eval()
    if torch.cuda.is_available():
        vocoder = vocoder
    else:
        vocoder = vocoder
    return vocoder


def load_and_setup_model(model_name, parser, arguments, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):

    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args(args=arguments)

    if model_name == "WaveGlow":
        print("ARGS:")
        print(model_args)
        print(model_args.wn_channels)

    unk_args[:] = list(set(unk_args) & set(model_unk_args))
    model_config = models.get_model_config(model_name, model_args)

    print("MODEL CONFIG:")
    print(model_name)
    print(model_config)

    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        model = load_model_from_ckpt(checkpoint, device, ema, model)

    if model_name == "WaveGlow":
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

        model = model.remove_weightnorm(model)

    if amp:
        model.half()
    model.eval()
    return model.to(device)


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c: f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False,
                           load_pitch=False, p_arpabet=0.0):
    tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)

    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    # for t in fields['text']:
    #     print(tp.sequence_to_text(t.numpy()))

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    if load_mels:
        assert 'mel' in fields
        fields['mel'] = [
            torch.load(Path(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor(
            [t.size(0) for t in fields['mel']])

    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(Path(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor(
            [t.size(0) for t in fields['pitch']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(
                    batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches, order


def build_pitch_transformation(flatten=False, invert=False, amplify=False, shift=0.0):
    fun = 'pitch'
    if flatten:
        fun = f'({fun}) * 0.0'
    if invert:
        fun = f'({fun}) * -1.0'
    if amplify:
        ampl = amplify
        fun = f'({fun}) * {ampl}'
    if shift != 0.0:
        hz = shift
        fun = f'({fun}) + {hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


class MeasureTime(list):
    def __init__(self, *args, cuda=False, **kwargs):
        super(MeasureTime, self).__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime((sum(ab) for ab in zip(self, other)), cuda=False)


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(
        description='PyTorch FastPitch Inference', allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args(args=arguments)
    return parser, args, unk_args


def load_models(parser, arguments, args, unk_args):
    global denoiser, waveglow, generator, mbmelgan, device_used
    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, keep_ambiguous=True)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device_used = torch.device('cuda' if args.cuda else 'cpu')

    if args.fastpitch != 'SKIP':
        generator = load_and_setup_model(
            'FastPitch', parser, arguments, args.fastpitch, args.amp, device_used,
            unk_args=unk_args, forward_is_infer=False, ema=args.ema,
            jitable=args.torchscript)

        if args.torchscript:
            generator = torch.jit.script(generator)
    else:
        generator = None

    if args.waveglow != 'SKIP':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveglow = load_and_setup_model(
                'WaveGlow', parser, arguments, args.waveglow, args.amp, device_used,
                unk_args=unk_args, forward_is_infer=True, ema=args.ema)
        denoiser = Denoiser(waveglow).to(device_used)
    else:
        waveglow = None

    #load multi-band melgan model
    mbmelgan = load_mb_melgan(args.mbmelgan)

    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def split_sentences(text, maxLength):
    punctuations = set([i for i in string.punctuation] + ['“‘”’'])

    def isAlpha(c: str):
        return c.isalnum()

    def split_passages(text):
        return [e.strip() for e in re.split(r'[\n]+', text)]

    def split_text(text, regex):
        return [e.strip() + d for e, d in zip(re.split(regex, text), re.findall(regex, text)) if e]

    def combine_sentences(sentences: list, maxLength: int = 30) -> list:
        if len(sentences) <= 1:
            return sentences
        if len(sentences[0].split(" ")) > maxLength:
            return [sentences[0]] + combine_sentences(sentences[1:], maxLength=maxLength)

        if len((sentences[0] + sentences[1]).split(" ")) <= maxLength:
            return combine_sentences([sentences[0] + " " + sentences[1]]+sentences[2:], maxLength=maxLength)
        else:
            return [sentences[0]] + combine_sentences(sentences[1:], maxLength=maxLength)

    def split_long_sentences(sentences: list, maxLength: int = 30) -> list:
        sub_sentences = []
        for sentence in sentences:
            if len(sentence.split(" ")) > maxLength:
                sub_sentences.append(split_text(sentence,  r'[?!.,:;-]'))
            else:
                sub_sentences.append([sentence])
        return sub_sentences
    
    def get_pieces(passage: str, maxLength: int):
        sub_sentences = split_long_sentences(split_text(passage, r'[.!?]'), maxLength)
        combined_sub_sentences = [combine_sentences(
            i, maxLength) for i in sub_sentences]
        flat_list = []
        for sublist in combined_sub_sentences:
            for item in sublist:
                item_chars = set([i for i in item])
                if not punctuations.issuperset(item_chars) and any(map(isAlpha, item_chars)):
                    flat_list.append(item)
        return flat_list
    
    passages = split_passages(text)
    result = []
    breaks = []
    for passage in passages:
        temp = get_pieces(passage, maxLength)
        result += temp
        if len(breaks) > 0:
            breaks += [breaks[-1]+len(temp)]
        else:
            breaks += [len(temp)]
    return result, breaks[0:len(breaks)-1]


def infer_waveglow(texts, args, pace=1.0, flatten=False, invert=False, shift=0.0, logger=None):
    print("<<<<<< start inference")
    fields = {"text": texts}
    batches, order = prepare_input_sequence(
        fields, device_used, args.symbol_set, args.text_cleaners, args.batch_size,
        args.dataset_path, load_mels=(generator is None), p_arpabet=args.p_arpabet)
    gen_measures = MeasureTime(cuda=False)
    waveglow_measures = MeasureTime(cuda=False)

    audio_outputs = []
    gen_infer_perf = None
    waveglow_infer_perf = None

    for b in batches:
        if generator is None:
            mel, mel_lens = b['mel'], b['mel_lens']
        else:
            with torch.no_grad(), gen_measures:
                mel, mel_lens, *_ = generator.infer(
                    b['text'],
                    pace=pace,
                    speaker=args.speaker,
                    pitch_transform=lambda pitch, pitch_lens, mean, std: pitch *
                    (0 if flatten else -1 if invert else 1) +
                    (shift/std if shift != 0.0 else 0)
                )

            gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
        if waveglow is not None:
            with torch.no_grad(), waveglow_measures:
                audios = waveglow.infer(mel, sigma=args.sigma_infer)
                audios = denoiser(audios.float(),
                                    strength=args.denoising_strength
                                    ).squeeze(1)
            waveglow_infer_perf = (
                audios.size(0) * audios.size(1) / waveglow_measures[-1])

            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * args.stft_hop_length]

                if args.fade_out:
                    fade_len = args.fade_out * args.stft_hop_length
                    fade_w = torch.linspace(1.0, 0.0, fade_len)
                    audio[-fade_len:] *= fade_w.to(audio.device)

                audio = audio / torch.max(torch.abs(audio))
                audio_outputs.append(audio.numpy())
    order = list(order)
    audio_outputs = [audio_outputs[i]
                     for i in [order.index(i) for i in range(len(order))]]
    return (audio_outputs,
            args.sampling_rate,
            {"fp": (gen_infer_perf, gen_measures[-1]), "wg": (waveglow_infer_perf, waveglow_measures[-1])})


def infer_hifigan(texts, args, pace=1.0, flatten=False, invert=False, shift=0.0):
    logger.debug("<<<<<< start inference")
    fields = {"text": texts}
    batches, order = prepare_input_sequence(
        fields, device_used, args.symbol_set, args.text_cleaners, args.batch_size,
        args.dataset_path, load_mels=(generator is None), p_arpabet=args.p_arpabet)
    gen_measures = MeasureTime(cuda=False)

    audio_outputs = []
    gen_infer_perf = None

    for b in batches:
        if generator is None:
            mel, mel_lens = b['mel'], b['mel_lens']
        else:
            with torch.no_grad(), gen_measures:
                mel, mel_lens, *_ = generator.infer(
                    b['text'],
                    pace=pace,
                    speaker=args.speaker,
                    pitch_transform=lambda pitch, pitch_lens, mean, std: pitch *
                    (0 if flatten else -1 if invert else 1) +
                    (shift/std if shift != 0.0 else 0)
                )

            gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]

        if hifigan is not None:
            with torch.no_grad():
                y_g_hat = hifigan(mel)
                audios = y_g_hat.squeeze()
                audios = audios * MAX_WAV_VALUE
                if len(audios.size()) == 1:
                    audios = audios.unsqueeze(0)

                for i, audio in enumerate(audios):
                    audio = audio[:mel_lens[i].item() * args.stft_hop_length]

                    if args.fade_out:
                        fade_len = args.fade_out * args.stft_hop_length
                        fade_w = torch.linspace(1.0, 0.0, fade_len)
                        audio[-fade_len:] *= fade_w.to(audio.device)

                    audio = audio / torch.max(torch.abs(audio))
                    audio_outputs.append(audio.numpy())

    order = list(order)
    audio_outputs = [audio_outputs[i] for i in [order.index(i) for i in range(len(order))]]
    return (audio_outputs,
            args.sampling_rate,
            {"fp": (gen_infer_perf, gen_measures[-1])})


def infer_mb_melgan(texts, args, pace=1.0, flatten=False, invert=False, shift=0.0):
    logger.debug("<<<<<< start inference")
    fields = {"text": texts}
    batches, order = prepare_input_sequence(
        fields, device_used, args.symbol_set, args.text_cleaners, args.batch_size,
        args.dataset_path, load_mels=(generator is None), p_arpabet=args.p_arpabet)
    gen_measures = MeasureTime(cuda=False)

    audio_outputs = []
    gen_infer_perf = None

    for b in batches:
        if generator is None:
            mel, mel_lens = b['mel'], b['mel_lens']
        else:
            with torch.no_grad(), gen_measures:
                mel, mel_lens, *_ = generator.infer(
                    b['text'],
                    pace=pace,
                    speaker=args.speaker,
                    pitch_transform=lambda pitch, pitch_lens, mean, std: pitch *
                    (0 if flatten else -1 if invert else 1) +
                    (shift/std if shift != 0.0 else 0)
                )

            gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]

        if mbmelgan is not None:
            with torch.no_grad():
                MAX_WAV_VALUE = 32768.0
                with torch.no_grad():
                    mel = mel.detach()
                if len(mel.shape) == 2:
                    mel = mel.unsqueeze(0)
                mel = mel
                audio = mbmelgan.inference(mel)
                # For multi-band inference
                if hp.model.out_channels > 1:
                    pqmf = PQMF()
                    audio = pqmf.synthesis(audio).view(-1)  
                audio = audio.squeeze() # collapse all dimension except time axis
                audio = audio[:-(hp.audio.hop_length*10)]
                audio = MAX_WAV_VALUE * audio
                audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
                audio = audio.short()
                audio_outputs.append(audio.detach().numpy())

    order = list(order)
    audio_outputs = [audio_outputs[i] for i in [order.index(i) for i in range(len(order))]]
    return (audio_outputs,
            args.sampling_rate,
            {"fp": (gen_infer_perf, gen_measures[-1])})