import re

from vietnam_number.number2word import n2w
from .symbols import vietnamese_re

_ampm_re = re.compile(
    r'([0-9]|0[0-9]|1[0-9]|2[0-3]):?([0-5][0-9])?\s*([AaPp][Mm]\b)')

day_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_date_seperator = r'(\/|-|\.)'
_full_date_pattern = r'(ngày)?' + vietnamese_re + r'(\d{1,2})' + \
    _date_seperator + r'(\d{1,2})' + _date_seperator + \
    r'(\d{4})' + vietnamese_re
_full_range_date_pattern = r'(ngày)?' + vietnamese_re + r'(\d{1,2})(\-)(\d{1,2})' + \
    _date_seperator + r'(\d{1,2})' + _date_seperator + \
    r'(\d{4})' + vietnamese_re
_day_month_pattern = r'(ngày)?' + vietnamese_re + r'(\d{1,2})' + \
    _date_seperator + r'(\d{1,2})' + vietnamese_re
_range_day_month_pattern = r'(ngày)?' + vietnamese_re + r'(\d{1,2})(\-)(\d{1,2})' + \
    _date_seperator + r'(\d{1,2})' + vietnamese_re
_month_year_pattern = r'(tháng)?' + vietnamese_re + r'(\d{1,2})' + \
    _date_seperator + r'(\d{4})' + vietnamese_re
_range_month_year_pattern = r'(tháng)?' + vietnamese_re + r'(\d{1,2})(\-)(\d{1,2})' + \
    _date_seperator + r'(\d{4})' + vietnamese_re
_time_pattern = vietnamese_re + \
    r'(\d{1,2})(g|:|h)(\d{1,2})(p|m)?' + vietnamese_re
_full_time_pattern = vietnamese_re + \
    r'(\d{1,2})(g|:|h)(\d{1,2})(p|:|m)(\d{1,2})(s|g)?' + vietnamese_re


def _remove_prefix_zero(text):
    text = text.strip().rstrip()
    while len(text) > 0 and text[0] == '0':
        text = text[1:]
    return text if (len(text) > 0) else '0'


def _expand_full_date(match):
    prefix, space, day, seporator1, month, seporator2, year, suffix = match.groups(
        0)
    day = _remove_prefix_zero(day)
    month = _remove_prefix_zero(month)
    year = _remove_prefix_zero(year)
    month = _remove_prefix_zero(month)
    _day = int(day)
    _month = int(month)
    if (_month > 12 or _month < 1):
        return match.group(0)
    if (_day > day_in_month[_month-1] or _day < 1):
        return match.group(0)
    return ' ngày ' + n2w(day) + ' tháng ' + n2w(month) + ' năm ' + n2w(year) + suffix + ' '


def _expand_range_full_date(match):
    prefix, space, day_start, hypen, day_end, seporator1, month, seporator2, year, suffix = match.groups(
        0)
    day_start = _remove_prefix_zero(day_start)
    day_end = _remove_prefix_zero(day_end)
    month = _remove_prefix_zero(month)
    year = _remove_prefix_zero(year)
    month = _remove_prefix_zero(month)
    _day_start = int(day_start)
    _day_end = int(day_end)
    _month = int(month)
    if (_month > 12 or _month < 1):
        return match.group(0)
    if (_day_start > day_in_month[_month-1] or _day_start < 1 or _day_end > day_in_month[_month-1] or _day_end < 1):
        return match.group(0)
    return ' ngày ' + n2w(day_start) + ' đến ngày ' + n2w(day_end) + ' tháng ' + n2w(month) + ' năm ' + n2w(year) + suffix + ' '


def _expand_day_month(match):
    prefix, space, day, seporator1, month, suffix = match.groups(0)
    day = _remove_prefix_zero(day)
    month = _remove_prefix_zero(month)
    _day = int(day)
    _month = int(month)
    if (_month > 12 or _month < 1):
        return match.group(0)
    if (_day > day_in_month[_month-1] or _day < 1):
        return match.group(0)
    return ' ngày ' + n2w(day) + ' tháng ' + n2w(month) + suffix + ' '


def _expand_range_day_month(match):
    prefix, space, day_start, hypen, day_end, seporator1, month, suffix = match.groups(
        0)
    day_start = _remove_prefix_zero(day_start)
    day_end = _remove_prefix_zero(day_end)
    month = _remove_prefix_zero(month)
    _day_start = int(day_start)
    _day_end = int(day_end)
    _month = int(month)
    if (_month > 12 or _month < 1):
        return match.group(0)
    if (_day_start > day_in_month[_month-1] or _day_start < 1 or _day_end > day_in_month[_month-1] or _day_end < 1):
        return match.group(0)
    return ' ngày ' + n2w(day_start) + ' đến ngày ' + n2w(day_end) + ' tháng ' + n2w(month) + suffix + ' '


def _expand_month_year(match):
    prefix, space, month, seporator1, year, suffix = match.groups(0)
    month = _remove_prefix_zero(month)
    year = _remove_prefix_zero(year)
    _month = int(month)
    if (_month > 12 or _month < 1):
        return match.group(0)
    return ' tháng ' + n2w(month) + ' năm ' + n2w(year) + suffix + ' '


def _expand_range_month_year(match):
    prefix, space, month_start, hypen, month_end, seporator1, year, suffix = match.groups(
        0)
    month_start = _remove_prefix_zero(month_start)
    month_end = _remove_prefix_zero(month_end)
    year = _remove_prefix_zero(year)
    _month_start = int(month_start)
    _month_end = int(month_end)
    if (_month_start > 12 or _month_start < 1 or _month_end > 12 or _month_end < 1):
        return match.group(0)
    return ' tháng ' + n2w(month_start) + ' đến tháng ' + n2w(month_end) + ' năm ' + n2w(year) + suffix + ' '


def _expand_time(math):
    prefix, hour, seporator1, minute, suffix, ending_space = math.groups(0)
    hour = _remove_prefix_zero(hour)
    minute = _remove_prefix_zero(minute)
    return ' ' + n2w(hour) + ' giờ ' + n2w(minute) + ' phút' + ending_space + ' '


def _expand_full_time(math):
    prefix, hour, seporator1, minute, seporator2, second, suffix, ending_space = math.groups(
        0)
    hour = _remove_prefix_zero(hour)
    minute = _remove_prefix_zero(minute)
    second = _remove_prefix_zero(second)
    return ' ' + n2w(hour) + ' giờ ' + n2w(minute) + ' phút ' + n2w(second) + ' giây' + ending_space + ' '


def normalize_datestime(text):
    text = re.sub(_range_month_year_pattern,
                  _expand_range_month_year, text, flags=re.IGNORECASE)
    text = re.sub(_full_range_date_pattern,
                  _expand_range_full_date, text, flags=re.IGNORECASE)
    text = re.sub(_full_date_pattern, _expand_full_date,
                  text, flags=re.IGNORECASE)
    text = re.sub(_month_year_pattern, _expand_month_year,
                  text, flags=re.IGNORECASE)
    text = re.sub(_range_day_month_pattern,
                  _expand_range_day_month, text, flags=re.IGNORECASE)
    text = re.sub(_day_month_pattern, _expand_day_month,
                  text, flags=re.IGNORECASE)
    text = re.sub(_time_pattern, _expand_time, text, flags=re.IGNORECASE)
    text = re.sub(_full_time_pattern, _expand_full_time,
                  text, flags=re.IGNORECASE)
    return text
