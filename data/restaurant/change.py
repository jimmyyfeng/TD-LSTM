#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import sys

lines = open(sys.argv[1]).readlines()
fp = open(sys.argv[2], 'w')
for i in xrange(0, len(lines), 3):
    sentence, aspect, polarity = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
    if polarity == '0':
        polarity = 'neutral'
    elif polarity == '1':
        polarity = 'positive'
    else:
        polarity = 'negative'

    words = sentence.split()
    print words
    ind = words.index('$T$')
    tmp = []
    for i, word in enumerate(words[:ind], 0):
        tmp.append(word + '/' + str(ind - i))
    for i, word in enumerate(words[ind + 1:], 1):
        tmp.append(word + '/' + str(i))
    sentence = ' '.join(tmp)
    fp.write(aspect + '||' + polarity + '||' + sentence + '\n')
