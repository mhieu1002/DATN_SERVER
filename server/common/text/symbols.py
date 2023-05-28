
'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from .cmudict import valid_symbols
import unicodedata


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in valid_symbols]
vietnamese_re = r'([^0-9a-zA-Z_ỹỷỵỳựữửừứủụợỡởờớộỗổồốỏọịỉệễểềếẽẻẹặẵẳằắậẫẩầấảạươũĩđăýúùõôóòíìêéèãâàáÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ])'
vietnamese_without_num_re = r'([^a-zA-Z_ỹỷỵỳựữửừứủụợỡởờớộỗổồốỏọịỉệễểềếẽẻẹặẵẳằắậẫẩầấảạươũĩđăýúùõôóòíìêéèãâàáÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ])'
vietnamese_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYỹỷỵỳựữửừứủụợỡởờớộỗổồốỏọịỉệễểềếẽẻẹặẵẳằắậẫẩầấảạươũĩđăýúùõôóòíìêéèãâàáÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'

def get_symbols(symbol_set='english_basic'):
    if symbol_set == 'english_basic':
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_basic_lowercase':
        _pad = '_'
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_expanded':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = 'áçéêëñöøćž'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_punctuation + _math + _special +
                       _accented + _letters) + _arpabet
    if symbol_set == 'vietnamese_basic':
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _accented = unicodedata.normalize(
            "NFC", 'ỹỷỵỳựữửừứủụợỡởờớộỗổồốỏọịỉệễểềếẽẻẹặẵẳằắậẫẩầấảạươũĩđăýúùõôóòíìêéèãâàáÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ')
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _punctuation + _special + _accented + _letters)
    elif symbol_set == 'vietnamese_lowercase':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = unicodedata.normalize(
            "NFC", 'ỹỷỵỳựữửừứủụợỡởờớộỗổồốỏọịỉệễểềếẽẻẹặẵẳằắậẫẩầấảạươũĩđăýúùõôóòíìêéèãâàá')
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        symbols = list(_punctuation + _math + _special + _accented + _letters)
    elif symbol_set == 'vietnamese_expanded':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = unicodedata.normalize(
            "NFC", 'ỹỷỵỳựữửừứủụợỡởờớộỗổồốỏọịỉệễểềếẽẻẹặẵẳằắậẫẩầấảạươũĩđăýúùõôóòíìêéèãâàáÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ')
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_punctuation + _math + _special + _accented + _letters)
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))

    return symbols


def get_pad_idx(symbol_set='english_basic'):
    if symbol_set in {'english_basic', 'english_basic_lowercase', 'vietnamese_basic'}:
        return 0
    else:
        raise Exception("{} symbol set not used yet".format(symbol_set))
