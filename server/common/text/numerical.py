""" adapted from https://github.com/keithito/tacotron """

from vietnam_number.number2word import n2w, n2w_single
import re

from .symbols import vietnamese_re, vietnamese_set, vietnamese_without_num_re

_magnitudes = ['trillion', 'billion', 'million',
               'thousand', 'hundred', 'm', 'b', 't']
_magnitudes_key = {'m': 'million', 'b': 'billion', 't': 'trillion'}
_measurements = '(f|c|k|d|m)'
_measurements_key = {'f': 'fahrenheit',
                     'c': 'celsius',
                     'k': 'thousand',
                     'm': 'meters'}
_currency_key = {'\$': 'đô la Mỹ',
                 '£': 'bảng Anh',
                 '€': 'ơ rô',
                 '₩': 'uân',
                 '₫': 'đồng',
                 'usd': 'đô la Mỹ',
                 'euro': 'ơ rô',
                 'eur': 'ơ rô',
                 'vnd': 'đồng',
                 'đ': 'đồng',
                 }
_letter_key_vi = {
    'a': 'ây',
    'b': 'bê',
    'c': 'xê',
    'd': 'dê',
    'đ': 'đê',
    'f': 'ép',
    'g': 'gờ',
    'h': 'hát',
    'i': 'ai',
    'j': 'chây',
    'k': 'kây',
    'l': 'lờ',
    'm': 'em mờ',
    'n': 'en nờ',
    'o': 'ô',
    'p': 'pi',
    'q': 'kiu',
    'r': 'rờ',
    's': 'ét',
    't': 'ti',
    'u': 'du',
    'v': 'vi',
    'w': 'vê kép',
    'x': 'ít',
    'z': 'dét',
}
_letter_combine_re = '|'.join(_letter_key_vi.keys())
_quotes_symbol = r'(\"|\')?'
_space = r'(\s)'
_letter_re = r'(chữ|chữ cái|kí tự|ký tự)?' + _space + _quotes_symbol + \
    r'(' + _letter_combine_re + r')' + r"(.)?"
_true_letter_re = r'(chữ|chữ cái|kí tự|ký tự)' + _space + _quotes_symbol + \
    r'([A-Z])' + _quotes_symbol + vietnamese_re
_measurement_re = re.compile(
    r'([0-9\.\,]*[0-9]+(\s)?{}\b)'.format(_measurements), re.IGNORECASE)
_multiply_re = re.compile(r'(\b[0-9]+)(x)([0-9]+)')

_negative_symbol_re = r'(.)(-{1})?'
_normal_number_re = r'[\d]+'
_number_with_one_middle_space_re = r'[\d]+[\s]{1}[\d]{3}'
_number_with_one_dot_middle_re = r'[\d]+[.]{1}[\d]{3}'
_number_with_two_dot_middle_re = r'[\d]+[.]{1}[\d]{3}[.]{1}[\d]{3}'
_number_with_three_dot_middle_re = r'[\d]+[.]{1}[\d]{3}[.]{1}[\d]{3}[.]{1}[\d]{3}'
_float_number_re = r'[\d]+[,]{1}[\d]+'
_roman_number_re = r'(\b(?!LLC)(?=[MDCLXVI]+\b)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b)'

_phone_re = r'(((\+84|84|0|0084){1})(3|5|7|8|9))+([0-9]{8})'

_end_number_re = r'(-)?(' + _float_number_re + '|' + _number_with_three_dot_middle_re + '|' + _number_with_two_dot_middle_re + '|' + _number_with_one_dot_middle_re + '|' + \
    _number_with_one_middle_space_re + '|' + _normal_number_re + r')'
_number_re = _negative_symbol_re + r'(' + _float_number_re + '|' + _number_with_three_dot_middle_re + '|' + _number_with_two_dot_middle_re + '|' + _number_with_one_dot_middle_re + '|' + \
    _number_with_one_middle_space_re + '|' + _normal_number_re + r')'

_currency_combine_regex = '|'.join(_currency_key.keys())
_currency_vi_re = re.compile(
   vietnamese_without_num_re + r'(' + _currency_combine_regex + ')' + vietnamese_without_num_re, re.IGNORECASE)
_range_number_re = re.compile(
    _number_re + r'(-|\s\-\s)' + _end_number_re + vietnamese_without_num_re , re.IGNORECASE)
_special_ordinal_pattern = r'(thứ)+ (1|4)'
_currency_re = re.compile(
    r'([0-9\.\,]*[0-9]+)(?:[ ]?({})(?=[^a-zA-Z]|$))?([\$€£₩])'.format("|".join(_magnitudes)), re.IGNORECASE)


def _expand_currency(match):
    prefix, currency, suffix = match.groups(
        0)
    suffix = '' if suffix == 0 else suffix
    if (currency == 'Đ' and suffix == '.'):
        return match.group(0)
    if (currency.lower() == '$'):
        currency = _currency_key['\$']
    elif (currency.lower() in _currency_key.keys()):
        currency = _currency_key[currency.lower()]
    return prefix + currency + suffix


# def _expand_measurement(m):
#     _, number, measurement = re.split('(\d+(?:\.\d+)?)', m.group(0))
#     number = _inflect.number_to_words(number)
#     measurement = "".join(measurement.split())
#     measurement = _measurements_key[measurement.lower()]
#     return "{} {}".format(number, measurement)


def _expand_roman(m):
    # from https://stackoverflow.com/questions/19308177/converting-roman-numerals-to-integers-in-python
    roman_numerals = {'I': 1, 'V': 5, 'X': 10,
                      'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    num = m.group(0).strip().rstrip()
    for i, c in enumerate(num):
        if (i+1) == len(num) or roman_numerals[c] >= roman_numerals[num[i+1]]:
            result += roman_numerals[c]
        else:
            result -= roman_numerals[c]
    if (int(result) > 39):
        return num
    return str(result)


def _expand_number(match):
    prefix, negative_symbol, number = match.groups(0)
    if prefix in vietnamese_set:
        negative_symbol = '' if negative_symbol == 0 else negative_symbol
        return prefix + ' ' + negative_symbol + n2w(number) + ' '
    else:
        number = '-' + number if negative_symbol == '-' else number
        return prefix + ' ' + n2w(number) + ' '


def _expand_phone(match):
    text: str = match.group(0).strip().rstrip()
    return n2w_single(text)


def _expand_range(match):
    prefix1, negative_symbol1, number_start, space, negative_symbol2, number_end, ending = match.groups(
        0)
    prefix1 = '' if prefix1 == 0 else prefix1
    if prefix1 in vietnamese_set:
        return match.group(0)
    else:
        number_start = '-' + number_start if negative_symbol1 == '-' else number_start
        number_end = '-' + number_end if negative_symbol2 == '-' else number_end
        return prefix1 + n2w(number_start) + ' đến ' + n2w(number_end) + ending


def _expand_ordinal(match):
    number = match.group(0).split()[1]
    if number == '1':
        return 'thứ nhất'
    elif number == '4':
        return 'thứ tư'
    else:
        return 'thứ {}'.format(n2w(number))


def _not_roman_number(match):
    return match.group(0).lower()


def _expand_letter_vi(match):
    leading, space1, quote1, char, trailing = match.groups(0)
    leading = '' if leading == 0 else leading
    quote1 = '' if quote1 == 0 else quote1
    trailing = '' if trailing == 0 else trailing
    if trailing in vietnamese_set:
        return match.group(0)
    char = char.lower()
    if char in _letter_key_vi:
        return leading + ' ' + quote1 + _letter_key_vi[char] + trailing + ' '
    return match.group(0)


def normalize_numbers(text):
    text = re.sub(_special_ordinal_pattern, _expand_ordinal, text)
    text = re.sub(_currency_vi_re, _expand_currency, text)
    text = re.sub(_range_number_re, _expand_range, text)
    text = re.sub(_phone_re, _expand_phone, text)
    text = re.sub(_number_re, _expand_number, text)
    text = re.sub(_letter_re, _expand_letter_vi, text, flags=re.IGNORECASE)
    return text

def normalize_roman_numbers(text):
    text = re.sub(_true_letter_re, _not_roman_number, text)
    text = re.sub(_roman_number_re, _expand_roman, text)
    return text