import re

_no_period_re = re.compile(r'(No[.])(?=[ ]?[0-9])')
_percent_re = re.compile(r'([ ]?[%])')
_half_re = re.compile('([0-9]½)|(½)')
_url_re = re.compile(r'([a-zA-Z])\.(com|gov|org|vn|com.vn|edu.vn)')


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('ms', 'miss'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ('sen', 'senator'),
    ('etc', 'et cetera'),
]]


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations_vi = [
    ('...', ' vân vân'),
    ('v.v', ' vân vân'),
    ('v/v', 'về việc'),
    ('đ/c', 'địa chỉ'),
    ('k/g', 'kính gửi'),
    ('th/g', 'thân gửi'),
    ('ko', 'không'),
    ('bit', 'biết'),
    ('bik', 'biết'),
    ('logic', 'lô zíc'),
    ('kit', 'kít'),
    ('test', 'tét'),
    ('noel', 'nô en'),
]


def _expand_no_period(m):
    word = m.group(0)
    if word[0] == 'N':
        return 'Number'
    return 'number'


def _expand_percent(m):
    return ' percent'


def _expand_percent_vi(m):
    return ' phần trăm'


def _expand_half(m):
    word = m.group(1)
    if word is None:
        return 'half'
    return word[0] + ' and a half'


def _expand_urls(m):
    return f'{m.group(1)} dot {m.group(2)}'


def _expand_urls_vi(m):
    return f'{m.group(1)} chấm {m.group(2)}'


def normalize_abbreviations(text):
    text = re.sub(_no_period_re, _expand_no_period, text)
    text = re.sub(_percent_re, _expand_percent, text)
    text = re.sub(_half_re, _expand_half, text)
    text = re.sub('&', ' and ', text)
    text = re.sub('@', ' at ', text)
    text = re.sub(_url_re, _expand_urls, text)

    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def normalize_abbreviations_vi(text):
    text = re.sub(_percent_re, _expand_percent_vi, text)
    # text = re.sub(_half_re, _expand_half, text)
    text = re.sub('&', ' và ', text)
    text = re.sub('@', ' a còng ', text)
    text = re.sub(_url_re, _expand_urls_vi, text)

    for abbre, replacement in _abbreviations_vi:
        text = text.replace(abbre, replacement)
    return text
