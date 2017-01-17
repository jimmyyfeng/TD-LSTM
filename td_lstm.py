#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import numpy as np
import tensorflow as tf
from utils import load_w2v, batch_index, load_inputs_twitter, load_word_id_mapping


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 100, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 200, 'number of example per batch')
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
tf.app.flags.DEFINE_string('type', 'TD', 'model type: ''(default), TD or TC')


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

            self.x_bw = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.y_bw = tf.placeholder(tf.int32, [None, self.n_class])
            self.sen_len_bw = tf.placeholder(tf.int32, [None])

        with tf.name_scope('weights'):
            self.weights = {
                'softmax_bi_lstm': tf.get_variable(
                    name='bi_lstm_w',
                    shape=[2 * self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax_bi_lstm': tf.get_variable(
                    name='bi_lstm_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

    def bi_dynamic_lstm(self, inputs_fw, inputs_bw):
        """
        :params: self.x, self.x_bw, self.seq_len, self.seq_len_bw,
                self.weights['softmax_lstm'], self.biases['softmax_lstm']
        :return: non-norm prediction values
        """
        inputs_fw = tf.nn.dropout(inputs_fw, keep_prob=self.dropout_keep_prob)
        inputs_bw = tf.nn.dropout(inputs_bw, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('forward_lstm'):
            outputs_fw, state_fw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_fw,
                sequence_length=self.sen_len,
                dtype=tf.float32,
                scope='LSTM_fw'
            )
            batch_size = tf.shape(outputs_fw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
            output_fw = tf.gather(tf.reshape(outputs_fw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        with tf.name_scope('backward_lstm'):
            outputs_bw, state_bw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_bw,
                sequence_length=self.sen_len_bw,
                dtype=tf.float32,
                scope='LSTM_bw'
            )
            batch_size = tf.shape(outputs_bw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len_bw - 1)
            output_bw = tf.gather(tf.reshape(outputs_bw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        output = tf.concat(1, [output_fw, output_bw])  # batch_size * 2n_hidden
        predict = tf.matmul(output, self.weights['softmax_bi_lstm']) + self.biases['softmax_bi_lstm']
        return predict

    def run(self):
        inputs_fw = tf.nn.embedding_lookup(self.word_embedding, self.x)
        inputs_bw = tf.nn.embedding_lookup(self.word_embedding, self.x_bw)
        prob = self.bi_dynamic_lstm(inputs_fw, inputs_bw)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, self.y)) + sum(reg_loss)


        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            acc_ = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            summary_loss = tf.scalar_summary('loss', cost)
            summary_acc = tf.scalar_summary('acc', acc_)
            train_summary_op = tf.merge_summary([summary_loss, summary_acc])
            validate_summary_op = tf.merge_summary([summary_loss, summary_acc])
            test_summary_op = tf.merge_summary([summary_loss, summary_acc])
            import time
            timestamp = str(int(time.time()))
            _dir = 'logs/' + str(timestamp) + '_' + self.type_ + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
            train_summary_writer = tf.train.SummaryWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.train.SummaryWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.train.SummaryWriter(_dir + '/validate', sess.graph)

            tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y = load_inputs_twitter(
                FLAGS.train_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.type_
            )
            te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y = load_inputs_twitter(
                FLAGS.test_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.type_
            )

            init = tf.initialize_all_variables()
            sess.run(init)

            max_acc = 0.
            for i in xrange(self.n_iter):
                for train, _ in self.get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, self.batch_size, 1.0):
                    _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                acc, loss, cnt, summary  = 0., 0., 0, None
                for test, num in self.get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, 2000, 1.0):
                    _loss, _acc, summary = sess.run([cost, accuracy, test_summary_op], feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                print cnt
                print acc
                test_summary_writer.add_summary(summary, step)
                print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(step, loss / cnt, acc / cnt)
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

    def get_batch_data(self, x, sen_len, x_bw, sen_len_bw, y, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.x_bw: x_bw[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.sen_len_bw: sen_len_bw[index],
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
