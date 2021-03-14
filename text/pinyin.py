'''
Defines the set of symbols and utilities for Chinese Mandarin Pinyin manipulation.

Supports the following Pinyin-with-label string (Pinyin with prosodic structure labeling information):

  Chinese text:      妈妈#1当时#1表示#3，儿子#1开心得#2像花儿#1一样#4。
  Raw Pinyin:        ma1 ma1 dang1 shi2 biao3 shi4 er2 zi5 kai1 xin1 de5 xiang4 huar1 yi2 yang4
  Pinyin-with-label: ma1-ma1 dang1-shi2 biao3-shi4, er2-zi5 kai1-xin1-de5 / xiang4-huar1 yi2-yang4.

  Meaning of prosodic strucutue labeling tags:
    '-': tag within prosodic word separating syllables (韵律词内部，分割不同音节/单字，一个韵律词由一或多个词典词构成)
    ' ': tag between different prosodic words          (韵律词边界，分割不同韵律词，用空格' '作为边界是为了保持与英语单词边界一致)
    '/': tag between different prosodic phrases        (韵律短语边界，分割不同韵律短语，相比英文增加了'/'边界)
    ',': tag between different intonational phrases    (语调短语边界，通常和子句对应，用逗号','保持与英语子句边界一致)
    '.': tag between different sentences               (语句边界，一句话结束，用句号'.'保持与英语句子边界一致)

  Author: johnson.tsing@gmail.com
'''


# Mandarin initials (普通话声母列表)
_initials = ['b', 'p', 'f', 'm', \
             'd', 't', 'n', 'l', \
             'g', 'k', 'h', \
             'j', 'q', 'x', \
             'zh', 'ch', 'sh', 'r', \
             'z', 'c', 's']

# Mandarin finals (普通话韵母列表)
_finals = ['a',  'ai', 'ao',  'an',  'ang', \
           'o',  'ou', 'ong', \
           'e',  'ei', 'en',  'eng', 'er', 'ev', \
           'i',  'ix', 'iy', \
           'ia', 'iao','ian', 'iang','ie', \
           'in', 'ing','io',  'iou', 'iong', \
           'u',  'ua', 'uo',  'uai', 'uei', \
           'uan','uen','uang','ueng', \
           'v',  've', 'van', 'vn', \
           'ng', 'mm', 'nn']

# Retroflex (Erhua) (儿化音信息)
_retroflex = ['rr']

# Tones (声调信息)
_tones = ['1', '2', '3', '4', '5']

# Prosodic structure symbols (韵律结构标记)
_prosodic_struct = ['-', ' ', '/', ',', '.', '?', '!']


def split_pinyin(pinyin):
  '''
  Split a single Pinyin into initial, final, tone and erhua (Retroflex). (With reference to pinyin.pl)

            tone - '1-4' for Chinese four tones, and '5' for neutral tone
       retroflex - 'rr' if Pinyin is Erhua (Retroflex), else ''
  initial, final - some special cases must be considered, and initial might be ''

  Args:
    - pinyin: Pinyin to be splitted

  Returns:
    - (initial, final, retroflex, tone)
	'''

  # retrieve tone
  input = pinyin
  if pinyin[-1] >= '0' and pinyin[-1] <= '9':
    tone = pinyin[-1]
    if tone == '0':
      tone = '5'
    pinyin = pinyin[:-1]
  else:
    tone = '5'

  # check Erhua (retroflex): Pinyin is not "er" and ending with "r"
  retroflex = ''
  if pinyin != 'er' and pinyin[-1] == 'r':
    pinyin = pinyin[:-1]
    retroflex = 'rr'

  # get initial, final
  initial = ''
  if pinyin[0] == 'y':
    # ya->ia, yan->ian, yang->iang, yao->iao, ye->ie, yo->io, yong->iong, you->iou
    # yi->i, yin->in, ying->ing
    # yu->v, yuan->van, yue->ve, yun->vn
    pinyin = 'i' + pinyin[1:]
    if pinyin[1] == 'i':
      pinyin = pinyin[1:]
    elif pinyin[1] in 'uv':
      pinyin = 'v' + pinyin[2:]
    final = pinyin
  elif pinyin[0] == 'w':
    # wa->ua, wo->uo, wai->uai, wei->uei, wan->uan, wen->uen, wang->uang, weng->ueng
    # wu->u
    # change 'w' to 'u', except 'wu->u'
    pinyin = 'u' + pinyin[1:]
    if pinyin[1] == 'u':
      pinyin = pinyin[1:]
    final = pinyin
  elif pinyin in ['ng', 'm', 'n']:
    # ng->ng, n->n, m->m
    final = pinyin
  else:
    # get initial and final
    # initial should be: b p m f d t n l g k h j q x z c s r zh ch sh
    final = pinyin
    if len(pinyin) > 1 and pinyin[:2] in ['ch', 'sh', 'zh']:
      initial = pinyin[:2]
      final = pinyin[2:]
    elif pinyin[0] in 'bpmfdtnlgkhjqxzcsr':
      initial = pinyin[:1]
      final = pinyin[1:]

    # the final of "zi, ci, si" should be "ix"
    if initial in ['c', 's', 'z'] and final == 'i':
      final = 'ix'
    # the final of "zhi, chi, shi, ri" should be "iy"
    elif initial in ['ch', 'r', 'sh', 'zh'] and final == 'i':
      final = 'iy'
    # ju->jv, jue->jve, juan->jvan, jun->jvn,
    # qu->qv, que->qve, quan->qvan, qun->qvn,
    # xu->xv, xue->xve, xuan->xvan, xun->xvn
    # change all leading 'u' to 'v'
    elif initial in ['j', 'q', 'x'] and final[0] == 'u':
      final = 'v' + final[1:]
    # lue->lve, nue->nve
    if final == 'ue':
      final ='ve'
    # ui->uei
    # iu->iou
    # un->uen
    if final == 'ui':
      final = 'uei'
    elif final == 'iu':
      final = 'iou'
    elif final == 'un':
      final = 'uen'
  # special process for final "ng, m, n, ev"
  # full pinyin might be "hng", "hm", "ng", "m", "n"
  # as there are to few samples, treat final "m, n" as initial, and "ng" as "n", and "ev" as "ei"
  if final == 'ng':
    final = 'n'
  if final == 'ev':
    final = 'ei'

  # keep the original input, if it is not Pinyin
  if (len(initial) and initial not in symbols) or (len(final) and final not in symbols):
    final = input
    initial = ''
    retroflex = ''
    tone = ''

  return (initial, final, retroflex, tone)


def pinyin_to_symbols(text):
  '''
  Convert Pinyin string to symbols.

  The input Pinyin string can contain optional prosodic structure labeling tags.
  Both following examples are OK:

    Raw Pinyin:        ma1 ma1 dang1 shi2 biao3 shi4 er2 zi5 kai1 xin1 de5 xiang4 huar1 yi2 yang4
    Pinyin-with-label: ma1-ma1 dang1-shi2 biao3-shi4, er2-zi5 kai1-xin1-de5 / xiang4-huar1 yi2-yang4.

  Args:
    - text: Pinyin string with optional prosodic structure labeling tags

  Returns:
    - List of symbols
  '''

  # ensure space between valid symbols
  text = text.replace(' ', ' | ')
  text = text.replace('/', ' / ')
  text = text.replace(',', ' , ')
  text = text.replace('.', ' . ')
  text = text.replace('?', ' ? ')
  text = text.replace('!', ' ! ')
  text = text.replace('-', ' - ')

  # split into tokens
  tokens = text.strip().split()

  # convert to symbols
  symbols = []
  for token in tokens:
    if token == '|':
      symbols.append(' ')
    elif token in ['-', '/', ',', '.', '?', '!']:
      symbols.append(token)
    else:
      (initial, final, retroflex, tone) = split_pinyin(token)
      if len(initial)>0:   symbols.append(initial)
      if len(final)>0:     symbols.append(final)
      if len(retroflex)>0: symbols.append(retroflex)
      if len(tone)>0:      symbols.append(tone)
  
  return symbols


# valid symbols for Chinese pinyin
symbols = _initials + _finals + _retroflex + _tones + _prosodic_struct
