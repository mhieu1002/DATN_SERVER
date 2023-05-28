import os
import logging
from logging.config import dictConfig
from flask.wrappers import Response
import torch
from flask import Flask, jsonify, request, make_response, abort
from io import BytesIO
import numpy as np
from inference import parse_arguments, load_models, infer_hifigan, split_sentences
from scipy.io.wavfile import write
from flask_cors import CORS
from common.text.text_processing import TextProcessing
from pydub import AudioSegment
import re

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)d} %(levelname)s - %(message)s',
        }
    },
    'handlers': {
        'default': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
        }
    },
    'loggers': {
        'myapp': {
            'handlers': ["default"],
            'level': 'DEBUG',
        },
    }
})

logger = logging.getLogger("myapp")

app = Flask(__name__)
CORS(app)
MAX_LENGTH = 30

device = torch.device("cpu")
arguments = [
    '-i',
    'vi_phrases/devset.tsv',
    '-o',
    './vi_output/audio',
    '--fastpitch',
    'models/FastPitch_checkpoint_1100.pt',
    '--hifigan',
    'models/g_00255000',
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


@app.after_request
def after_request(response: Response):
    response.cache_control.max_age = 0
    response.cache_control.no_cache = True
    return response


def getResult(text: str, pace, flatten, invert, shift):
    text_cleaned = tp.clean_text(text)
    if text_cleaned[-1] not in '?.!':
        text_cleaned += '.'
    text_cleaned = re.sub("(?<![.!?])[\n]+", ".\n", text_cleaned)
    sentences, breaks = split_sentences(text_cleaned, MAX_LENGTH)
    result = infer_hifigan(sentences, args, pace=pace,
                               flatten=flatten, invert=invert, shift=shift)
    return (result, breaks)


@app.route('/tts-predict', methods=['POST'])
def tts():
    json = request.get_json()
    texts = json['texts']
    try:
        pace = float(json['pace'])
    except (KeyError):
        pace = 1.0
    try:
        flatten = json['flatten']
    except (KeyError):
        flatten = False
    try:
        invert = json['invert']
    except (KeyError):
        invert = False
    try:
        shift = json['shift']
    except (KeyError):
        shift = 0
    try:
        assert(type(flatten) is bool)
        assert(type(invert) is bool)
        assert(type(shift) is int)
        assert(pace > 0.0)
    except (AssertionError):
        abort(Response(status=400,
                       response="""Please provide pace, flatten, invert and shift values, like this:
        {
        "texts": [
            "tôi thích đạp xe vào buổi chiều"
        ], // <String>[]
        "pace": 1.0, // unsigned float, must be greater than 0.0
        "flatten": false, // boolean
        "invert": true, // boolean
        "shift": -50 // signed int
        }"""))
    buffer = BytesIO()
    buffer.flush()
    result, breaks = getResult(texts[0], pace, flatten, invert, shift)
    audios, rate, perf = result
    audios = list(audios)
    for i, _break in enumerate(breaks):
        audios.insert(_break+i, list(AudioSegment.silent(500, 22050).raw_data))
    audio = np.concatenate(audios)
    audio /= (1.414 / 32767)
    audio = audio.astype(np.int16)
    write(buffer, rate, audio)
    response = make_response(buffer.getvalue())
    buffer.close()
    response.headers['FastPitch-Latency'] = perf['fp'][1]
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = f'attachment; filename=audio.wav'
    return response


@app.route('/tts-process', methods=['POST'])
def process():
    json = request.get_json()
    texts = json['texts']
    result = []
    for text in texts:
        result.append({
            'data_raw': text,
            'processed': tp.clean_text(text),
        })
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))