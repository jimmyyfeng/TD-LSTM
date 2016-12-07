#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import numpy as np
import tensorflow as tf
from utils import load_w2v, batch_index, load_inputs_twitter, load_word_id_mapping


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 100, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 100, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 10, 'number of train iter')

tf.app.flags.DEFINE_string('train_file_path', 'data/train.raw', 'training file')
tf.app.flags.DEFINE_string('validate_file_path', 'data/validate.raw', 'validating file')
tf.app.flags.DEFINE_string('test_file_path', 'data/test.raw', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter_word_embedding_partial_100.txt', 'embedding file')
tf.app.flags.DEFINE_string('word_id_file_path', 'data/word_id.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('type', '', 'model type: ''(default), TD or TC')


class LSTM(object):

    def __init__(self, embedding_dim=100, batch_size=64, n_hidden=100, learning_rate=0.01,
                 n_class=3, max_sentence_len=50, l2_reg=0., display_step=4, n_iter=100, type_=''):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.display_step = display_step
        self.n_iter = n_iter
        self.type_ = type_
        self.word_id_mapping, self.w2v = load_w2v(FLAGS.embedding_file_path, self.embedding_dim)
        self.word_embedding = tf.constant(self.w2v, name='word_embedding')
        # self.word_embedding = tf.Variable(self.w2v, name='word_embedding')
        # self.word_id_mapping = load_word_id_mapping(FLAGS.word_id_file_path)
        # self.word_embedding = tf.Variable(
        #     tf.random_uniform([len(self.word_id_mapping), self.embedding_dim], -0.1, 0.1), name='word_embedding')

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.y = tf.placeholder(tf.int32, [None, self.n_class])
            self.sen_len = tf.placeholder(tf.int32, None)

        with tf.name_scope('weights'):
            self.weights = {
                'softmax_lstm': tf.get_variable(
                    name='lstm_w',
                    shape=[self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax_lstm': tf.get_variable(
                    name='lstm_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

    def dynamic_lstm(self, inputs):
        """
        :params: self.x, self.seq_len, self.weights['softmax_lstm'], self.biases['softmax_lstm']
        :return: non-norm prediction values
        """
        inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('dynamic_rnn'):
            outputs, state = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs,
                sequence_length=self.sen_len,
                dtype=tf.float32,
                scope='LSTM'
            )
            batch_size = tf.shape(outputs)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
            output = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)  # batch_size * n_hidden
        predict = tf.matmul(output, self.weights['softmax_lstm']) + self.biases['softmax_lstm']

        return predict

    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        prob = self.dynamic_lstm(inputs)

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, self.y))

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        with tf.Session() as sess:
            summary_loss = tf.scalar_summary('loss', cost)
            summary_acc = tf.scalar_summary('acc', accuracy)
            train_summary_op = tf.merge_summary([summary_loss, summary_acc])
            validate_summary_op = tf.merge_summary([summary_loss, summary_acc])
            test_summary_op = tf.merge_summary([summary_loss, summary_acc])
            import time
            timestamp = str(int(time.time()))
            _dir = 'logs/' + str(timestamp) + '_' + self.type_ + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
            train_summary_writer = tf.train.SummaryWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.train.SummaryWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.train.SummaryWriter(_dir + '/validate', sess.graph)

            tr_x, tr_sen_len, tr_y = load_inputs_twitter(
                FLAGS.train_file_path,
                self.word_id_mapping,
                self.max_sentence_len
            )
            te_x, te_sen_len, te_y = load_inputs_twitter(
                FLAGS.test_file_path,
                self.word_id_mapping,
                self.max_sentence_len
            )

            init = tf.initialize_all_variables()
            sess.run(init)

            max_acc = 0.
            for i in xrange(self.n_iter):
                for train, _ in self.get_batch_data(tr_x, tr_y, tr_sen_len, self.batch_size, 1.0):
                    _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                acc, loss, cnt = 0., 0., 0
                for test, num in self.get_batch_data(te_x, te_y, te_sen_len, 2000, 1.0):
                    _loss, _acc, summary = sess.run([cost, accuracy, test_summary_op], feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                print cnt
                print acc
                test_summary_writer.add_summary(summary, step)
                print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(step, loss / cnt, acc / cnt)
                test_summary_writer.add_summary(summary, step)
                if acc / cnt > max_acc:
                    max_acc = acc / cnt
            print 'Optimization Finished! Max acc={}'.format(max_acc)

            print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
                self.learning_rate,
                self.n_iter,
                self.batch_size,
                self.n_hidden,
                self.l2_reg
            )

    def get_batch_data(self, x, y, sen_len, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)


def main(_):
    lstm = LSTM(
        embedding_dim=FLAGS.embedding_dim,
        batch_size=FLAGS.batch_size,
        n_hidden=FLAGS.n_hidden,
        learning_rate=FLAGS.learning_rate,
        n_class=FLAGS.n_class,
        max_sentence_len=FLAGS.max_sentence_len,
        l2_reg=FLAGS.l2_reg,
        display_step=FLAGS.display_step,
        n_iter=FLAGS.n_iter,
        type_=FLAGS.type
    )
    lstm.run()


if __name__ == '__main__':
    tf.app.run()
