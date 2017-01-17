#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import numpy as np


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = range(length)
    for j in xrange(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in xrange(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print '\nload word-id mapping done!\n'
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print 'a bad word embedding: {}'.format(line[0])
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print np.shape(w2v)
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print word_dict['$t$'], len(w2v)
    return word_dict, w2v


def load_word_embedding(word_id_file, w2v_file, embedding_dim, is_skip=False):
    word_to_id = load_word_id_mapping(word_id_file)
    word_dict, w2v = load_w2v(w2v_file, embedding_dim, is_skip)
    cnt = len(w2v)
    for k in word_to_id.keys():
        if k not in word_dict:
            word_dict[k] = cnt
            w2v = np.row_stack((w2v, np.random.uniform(-0.01, 0.01, (embedding_dim,))))
            cnt += 1
    print len(word_dict), len(w2v)
    return word_dict, w2v


def load_aspect2id(input_file, word_id_mapping, w2v, embedding_dim):
    aspect2id = dict()
    a2v = list()
    a2v.append([0.] * embedding_dim)
    cnt = 0
    for line in open(input_file):
        line = line.lower().split()
        cnt += 1
        aspect2id[' '.join(line[:-1])] = cnt
        tmp = []
        for word in line:
            if word in word_id_mapping:
                tmp.append(w2v[word_id_mapping[word]])
        if tmp:
            a2v.append(np.sum(tmp, axis=0) / len(tmp))
        else:
            a2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))
    print len(aspect2id), len(a2v)
    return aspect2id, np.asarray(a2v, dtype=np.float32)


def change_y_to_onehot(y):
    from collections import Counter
    print Counter(y)
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    lines = open(input_file).readlines()
    for i in xrange(0, len(lines), 3):
        target_word = lines[i + 1].decode(encoding).lower().split()
        target_word = map(lambda w: word_to_id.get(w, 0), target_word)
        target_words.append([target_word[0]])

        y.append(lines[i + 2].strip().split()[0])

        words = lines[i].decode(encoding).lower().split()
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            words_l.extend(target_word)
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            tmp = target_word + words_r
            tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
        else:
            words = words_l + target_word + words_r
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))

    y = change_y_to_onehot(y)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y), np.asarray(target_words)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def extract_aspect_to_id(input_file, aspect2id_file):
    dest_fp = open(aspect2id_file, 'w')
    lines = open(input_file).readlines()
    targets = set()
    for i in xrange(0, len(lines), 3):
        target = lines[i + 1].lower().split()
        targets.add(' '.join(target))
    aspect2id = list(zip(targets, range(1, len(lines) + 1)))
    for k, v in aspect2id:
        dest_fp.write(k + ' ' + str(v) + '\n')


def load_inputs_twitter_at(input_file, word_id_file, aspect_id_file, sentence_len, type_='', encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'
    if type(aspect_id_file) is str:
        aspect_to_id = load_aspect2id(aspect_id_file)
    else:
        aspect_to_id = aspect_id_file
    print 'load aspect-to-id done!'

    x, y, sen_len = [], [], []
    aspect_words = []
    lines = open(input_file).readlines()
    for i in xrange(0, len(lines), 3):
        aspect_word = ' '.join(lines[i + 1].lower().split())
        aspect_words.append(aspect_to_id.get(aspect_word, 0))

        y.append(lines[i + 2].split()[0])

        words = lines[i].decode(encoding).lower().split()
        ids = []
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
        # ids = list(map(lambda word: word_to_id.get(word, 0), words))
        sen_len.append(len(ids))
        x.append(ids + [0] * (sentence_len - len(ids)))
    cnt = 0
    for item in aspect_words:
        if item > 0:
            cnt += 1
    print 'cnt=', cnt
    y = change_y_to_onehot(y)
    for item in x:
        if len(item) != sentence_len:
            print 'aaaaa=', len(item)
    x = np.asarray(x, dtype=np.int32)

    return x, np.asarray(sen_len), np.asarray(aspect_words), np.asarray(y)


