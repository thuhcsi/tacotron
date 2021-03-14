""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.

For Chinese Mandarin, the set of symbols can be switched to Pinyin initials, finals, retroflex (Erhua), 
tones and prosodic structure tags.

The set of symbols can be switched according to language tag in symbols(lang).
'''

from . import cmudict
from . import pinyin

_pad        = '_'
_eos        = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? %/'
_digits     = '0123456789'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# English characters
en_symbols = [_pad, _eos] + list(_characters) + list(_digits) #+ _arpabet

# Chinese Pinyin symbols (intitial, final, tone, etc)
zh_py_symbols = [_pad, _eos] + pinyin.symbols

# Export symbols according to language tag
def symbols(lang):
  if lang == 'py':
    return zh_py_symbols
  elif lang == 'en':
    return en_symbols
  else:
    raise NameError('Unknown target language: %s' % str(lang))
