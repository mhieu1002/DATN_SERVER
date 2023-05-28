""" adapted from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
"""


# Regular expression matching whitespace:
import re
from unidecode import unidecode
import unicodedata

from vi_cleaner.vi_cleaner import ViCleaner
from .numerical import normalize_numbers, normalize_roman_numbers
from .acronyms import normalize_acronyms, spell_acronyms, spell_acronyms_vi
from .datestime import normalize_datestime
from .letters_and_numbers import normalize_letters_and_numbers_vi, normalize_letters_and_numbers
from .abbreviations import normalize_abbreviations, normalize_abbreviations_vi

_whitespace_re = re.compile(r"\s+")


def expand_abbreviations(text):
    return normalize_abbreviations(text)


def expand_abbreviations_vi(text):
    return normalize_abbreviations_vi(text)


def expand_numbers(text):
    return normalize_numbers(text)


def expand_roman_numbers_vi(text):
    return normalize_roman_numbers(text)


def expand_acronyms(text):
    return normalize_acronyms(text)


def expand_datestime(text):
    return normalize_datestime(text)


def expand_letters_and_numbers(text):
    return normalize_letters_and_numbers(text)


def expand_letters_and_numbers_vi(text):
    return normalize_letters_and_numbers_vi(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def separate_acronyms(text):
    text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
    return text


def convert_to_ascii(text):
    return unidecode(text)


def normalize_ascii_vi(text):
    return unicodedata.normalize("NFC", text)


def basic_cleaners(text):
    """Basic pipeline that collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, with number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners_v2(text):
    text = convert_to_ascii(text)
    text = expand_datestime(text)
    text = expand_letters_and_numbers(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = spell_acronyms(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    # compatibility with basic_english symbol set
    text = re.sub(r"/+", " ", text)
    return text


def vietnamese_cleaners(text):
    """Pipeline for English text, with number and abbreviation expansion."""
    text = vietnamese_cleaners_v2(text)
    return text


def vietnamese_cleaners_v2(text):
    viCleaer = ViCleaner(text)
    text = viCleaer.clean()
    # text = ' ' + text + ' '
    # text = text.replace('\n', ' ')
    # text = normalize_ascii_vi(text)
    # text = expand_abbreviations_vi(text)
    # text = expand_roman_numbers_vi(text)
    # text = spell_acronyms_vi(text)
    # text = expand_datestime(text)
    # text = expand_letters_and_numbers_vi(text)
    # text = expand_numbers(text)
    # text = lowercase(text)
    # text = re.sub(r'(\s)?(/){1}(\s)?', ' trên ', text)
    # text = re.sub(r'(\s)?(\+){1}(\s)?', ' cộng ', text)
    # text = re.sub(r'(-)', ' ', text)
    # text = collapse_whitespace(text)
    return text
