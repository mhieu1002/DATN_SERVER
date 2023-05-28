from random import sample
import re

from vietnam_number.number2word import n2w
from .symbols import vietnamese_re, vietnamese_without_num_re

_letters_and_numbers_re = re.compile(
    r"((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9']*)", re.IGNORECASE)

_hardware_re = re.compile(
    '([0-9]+(?:[.,][0-9]+)?)(?:\s?)(tb|gb|mb|kb|ghz|mhz|khz|hz|mm)', re.IGNORECASE)
_hardware_key = {'tb': 'terabyte',
                 'gb': 'gigabyte',
                 'mb': 'megabyte',
                 'kb': 'kilobyte',
                 'ghz': 'gigahertz',
                 'mhz': 'megahertz',
                 'khz': 'kilohertz',
                 'hz': 'hertz',
                 'mm': 'millimeter',
                 'cm': 'centimeter',
                 'km': 'kilometer'}
_measurement_key_vi = {
    'p': 'phút',
    'tb': 'tê ra bai',
    'gb': 'gi ga bai',
    'mb': 'mê ga bai',
    'kb': 'ki lô bai',
    'ghz': 'gi ga héc',
    'mhz': 'mê ga héc',
    'khz': 'ki lô héc',
    'hz': 'héc',
    'm2': 'mét vuông',
    'km2': 'ki lô mét vuông',
    'm3': 'mét khối',
    'km3': 'ki lô mét khối',
    'nm': 'na nô mét',
    'mm': 'mi li mét',
    'cm': 'xen ti mét',
    'dm': 'đề xi mét',
    'dam': 'đề ca mét',
    'hm': 'héc tô mét',
    'km': 'ki lô mét',
    'kg': 'ki lô gam',
    'hg': 'héc tô gam',
    'dag': 'đề ca gam',
    'mg': 'mi li gam',
    'ml': 'mi li lít',
    'l': 'lít',
    'g': 'gam',
    'm': 'mét',
    '"': 'inch',
    'in': 'inch',
    'h': 'giờ',
    'ha': 'héc ta',
}

_measurement_combine_regex = '|'.join(_measurement_key_vi.keys())
_measurement_pattern = re.compile(
    vietnamese_without_num_re + '(' + _measurement_combine_regex + ')' + vietnamese_re, re.IGNORECASE)
# _measurement_range_pattern = re.compile(
#     _number_re + r'(-)' + _end_number_re + r'(\s)?' + '(' + _measurement_combine_regex + ')' + _vietnamese_re, re.IGNORECASE)

_dimension_re = re.compile(
    r'\b(\d+(?:[,.]\d+)?\s*[xX]\s*\d+(?:[,.]\d+)?\s*[xX]\s*\d+(?:[,.]\d+)?(?:in|inch|m)?)\b|\b(\d+(?:[,.]\d+)?\s*[xX]\s*\d+(?:[,.]\d+)?(?:in|inch|m)?)\b')
_dimension_key = {'m': 'meter',
                  'in': 'inch',
                  'inch': 'inch'}


def _expand_letters_and_numbers(m):
    text = re.split(r'(\d+)', m.group(0))

    # remove trailing space
    if text[-1] == '':
        text = text[:-1]
    elif text[0] == '':
        text = text[1:]

    # if not like 1920s, or AK47's , 20th, 1st, 2nd, 3rd, etc...
    if text[-1] in ("'s", "s", "th", "nd", "st", "rd") and text[-2].isdigit():
        text[-2] = text[-2] + text[-1]
        text = text[:-1]

    # for combining digits 2 by 2
    new_text = []
    for i in range(len(text)):
        string = text[i]
        if string.isdigit() and len(string) < 5:
            # heuristics
            if len(string) > 2 and string[-2] == '0':
                if string[-1] == '0':
                    string = [string]
                else:
                    string = [string[:-2], string[-2], string[-1]]
            elif len(string) % 2 == 0:
                string = [string[i:i+2] for i in range(0, len(string), 2)]
            elif len(string) > 2:
                string = [string[0]] + [string[i:i+2]
                                        for i in range(1, len(string), 2)]
            new_text.extend(string)
        else:
            new_text.append(string)

    text = new_text
    text = " ".join(text)
    return text


def _expand_hardware(m):
    quantity, measure = m.groups(0)
    measure = _hardware_key[measure.lower()]
    if measure[-1] != 'z' and float(quantity.replace(',', '')) > 1:
        return "{} {}s".format(quantity, measure)
    return "{} {}".format(quantity, measure)


def _expand_measurement_vi(match):
    prefix, measure, suffix = match.groups(0)
    measure = measure.lower()
    if (measure == 'h' and not prefix.isdigit()):
        return match.group(0)
    measure = _measurement_key_vi[measure]
    return prefix + ' ' + measure + ' ' + suffix


def _expand_measurement_range_vi(match):
    start, hyphen, end, space, measure, trailing_char = match.groups(0)
    quantity = n2w(start) + ' đến ' + n2w(end)
    measure = _measurement_key_vi[measure.lower().strip().rstrip()]
    return "{} {} ".format(quantity, measure)


def _expand_dimension(m):
    text = "".join([x for x in m.groups(0) if x != 0])
    text = text.replace(' x ', ' by ')
    text = text.replace('x', ' by ')
    if text.endswith(tuple(_dimension_key.keys())):
        if text[-2].isdigit():
            text = "{} {}".format(text[:-1], _dimension_key[text[-1:]])
        elif text[-3].isdigit():
            text = "{} {}".format(text[:-2], _dimension_key[text[-2:]])
    return text


def normalize_letters_and_numbers(text):
    text = re.sub(_hardware_re, _expand_hardware, text)
    text = re.sub(_dimension_re, _expand_dimension, text)
    text = re.sub(_letters_and_numbers_re, _expand_letters_and_numbers, text)
    return text


def normalize_letters_and_numbers_vi(text):
    text = re.sub(_measurement_pattern, _expand_measurement_vi, text)
    # text = re.sub(_measurement_range_pattern,
    #               _expand_measurement_range_vi, text)
    # text = re.sub(_dimension_re, _expand_dimension, text)
    # text = re.sub(_letters_and_numbers_re, _expand_letters_and_numbers, text)
    return text
