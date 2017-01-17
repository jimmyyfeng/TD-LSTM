#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import tensorflow as tf
from utils import load_w2v, batch_index, load_word_embedding, load_aspect2id, load_inputs_twitter_at


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 25, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 20, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob')


tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/rest_2014_lstm_train_new.txt', 'training file')
tf.app.flags.DEFINE_string('validate_file_path', 'data/restaurant/rest_2014_lstm_test_new.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_2014_lstm_test_new.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/rest_2014_word_embedding_300_new.txt', 'embedding file')
tf.app.flags.DEFINE_string('word_id_file_path', 'data/restaurant/word_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('aspect_id_file_path', 'data/restaurant/aspect_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('t', 'last', 'model type: ')


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
        self.word_id_mapping, self.w2v = load_word_embedding(FLAGS.word_id_file_path, FLAGS.embedding_file_path, self.embedding_dim)
        # self.word_embedding = tf.constant(self.w2v, dtype=tf.float32, name='word_embedding')
        self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding')
        # self.word_id_mapping = load_word_id_mapping(FLAGS.word_id_file_path)
        # self.word_embedding = tf.Variable(
        #     tf.random_uniform([len(self.word_id_mapping), self.embedding_dim], -0.1, 0.1), name='word_embedding')
        self.aspect_id_mapping, self.aspect_embed = load_aspect2id(FLAGS.aspect_id_file_path, self.word_id_mapping, self.w2v, self.embedding_dim)
        self.aspect_embedding = tf.Variable(self.aspect_embed, dtype=tf.float32, name='aspect_embedding')

        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x')
            self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y')
            self.sen_len = tf.placeholder(tf.int32, None, name='sen_len')
            self.aspect_id = tf.placeholder(tf.int32, None, name='aspect_id')

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.get_variable(
                    name='softmax_w',
                    shape=[self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.get_variable(
                    name='softmax_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        self.W = tf.get_variable(
            name='W',
            shape=[self.n_hidden + self.embedding_dim, self.n_hidden + self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.w = tf.get_variable(
            name='w',
            shape=[self.n_hidden + self.embedding_dim, 1],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wp = tf.get_variable(
            name='Wp',
            shape=[self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wx = tf.get_variable(
            name='Wx',
            shape=[self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )

    def dynamic_rnn(self, cell, inputs, length, max_len, scope_name, out_type='all'):
        outputs, state = tf.nn.dynamic_rnn(
            cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        )  # outputs -> batch_size * max_len * n_hidden
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)  # batch_size * n_hidden
        elif out_type == 'all_avg':
            outputs = LSTM.reduce_mean(outputs, length)
        return outputs

    def bi_dynamic_rnn(self, cell, inputs, length, max_len, scope_name, out_type='all'):
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell(self.n_hidden),
            cell_bw=cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        )
        if out_type == 'last':
            outputs_fw, outputs_bw = outputs
            outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
            outputs = tf.concat(2, [outputs_fw, outputs_bw])
        else:
            outputs = tf.concat(2, outputs)  # batch_size * max_len * 2n_hidden
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, 2 * self.n_hidden]), index)  # batch_size * 2n_hidden
        elif out_type == 'all_avg':
            outputs = LSTM.reduce_mean(outputs, length)  # batch_size * 2n_hidden
        return outputs

    def AE(self, inputs, target, type_='last'):
        """
        :params: self.x, self.seq_len, self.weights['softmax_lstm'], self.biases['sof
        :return: non-norm prediction values
        """
        print 'I am AE.'
        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.embedding_dim])
        target = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * target
        inputs = tf.concat(2, [inputs, target])
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)

        cell = tf.nn.rnn_cell.LSTMCell
        outputs = self.dynamic_rnn(cell, inputs, self.sen_len, self.max_sentence_len, 'AE', FLAGS.t)

        return LSTM.softmax_layer(outputs, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    def AT(self, inputs, target, type_=''):
        print 'I am AT.'
        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.embedding_dim])
        target = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * target
        in_t = tf.concat(2, [inputs, target])
        in_t = tf.nn.dropout(in_t, keep_prob=self.keep_prob1)
        cell = tf.nn.rnn_cell.LSTMCell
        hiddens = self.dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT', 'all')

        h_t = tf.reshape(tf.concat(2, [hiddens, target]), [-1, self.n_hidden + self.embedding_dim])

        M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W)), self.w)
        alpha = LSTM.softmax(tf.reshape(M, [-1, 1, self.max_sentence_len]), self.sen_len, self.max_sentence_len)
        self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len])

        r = tf.reshape(tf.batch_matmul(alpha, hiddens), [-1, self.n_hidden])
        index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
        hn = tf.gather(tf.reshape(hiddens, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        h = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(hn, self.Wx))

        return LSTM.softmax_layer(h, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    @staticmethod
    def softmax_layer(inputs, weights, biases, keep_prob):
        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
            predict = tf.matmul(outputs, weights) + biases
            predict = tf.nn.softmax(predict)
        return predict

    @staticmethod
    def reduce_mean(inputs, length):
        """
        :param inputs: 3-D tensor
        :param length: the length of dim [1]
        :return: 2-D tensor
        """
        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
        return inputs

    @staticmethod
    def softmax(inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, 2, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        aspect = tf.nn.embedding_lookup(self.aspect_embedding, self.aspect_id)
        if FLAGS.method == 'AE':
            prob = self.AE(inputs, aspect, FLAGS.t)
        elif FLAGS.method == 'AT':
            prob = self.AT(inputs, aspect, FLAGS.t)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, self.y))
            cost = - tf.reduce_mean(tf.cast(self.y, tf.float32) * tf.log(prob)) + sum(reg_loss)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            true_y = tf.argmax(self.y, 1)
            pred_y = tf.argmax(prob, 1)
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            _acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
                FLAGS.keep_prob1,
                FLAGS.keep_prob2,
                FLAGS.batch_size,
                FLAGS.learning_rate,
                FLAGS.l2_reg,
                FLAGS.max_sentence_len,
                FLAGS.embedding_dim,
                FLAGS.n_hidden,
                FLAGS.n_class
            )
            summary_loss = tf.scalar_summary('loss' + title, cost)
            summary_acc = tf.scalar_summary('acc' + title, _acc)
            train_summary_op = tf.merge_summary([summary_loss, summary_acc])
            validate_summary_op = tf.merge_summary([summary_loss, summary_acc])
            test_summary_op = tf.merge_summary([summary_loss, summary_acc])
            import time
            timestamp = str(int(time.time()))
            _dir = 'logs/' + str(timestamp) + '_' + title
            train_summary_writer = tf.train.SummaryWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.train.SummaryWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.train.SummaryWriter(_dir + '/validate', sess.graph)

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            init = tf.initialize_all_variables()
            sess.run(init)

            # saver.restore(sess, 'models/logs/1481529975__r0.005_b2000_l0.05self.softmax/-1072')

            save_dir = 'models/' + _dir + '/'
            import os
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            tr_x, tr_sen_len, tr_target_word, tr_y = load_inputs_twitter_at(
                FLAGS.train_file_path,
                self.word_id_mapping,
                self.aspect_id_mapping,
                self.max_sentence_len,
                self.type_
            )
            te_x, te_sen_len, te_target_word, te_y = load_inputs_twitter_at(
                FLAGS.test_file_path,
                self.word_id_mapping,
                self.aspect_id_mapping,
                self.max_sentence_len,
                self.type_
            )

            max_acc = 0.
            max_alpha = None
            max_ty, max_py = None, None
            for i in xrange(self.n_iter):
                for train, _ in self.get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, self.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                    _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)

                acc, loss, cnt = 0., 0., 0
                flag = True
                summary, step = None, None
                alpha = None
                ty, py = None, None
                for test, num in self.get_batch_data(te_x, te_sen_len, te_y, te_target_word, 2000, 1.0, 1.0, False):
                    _loss, _acc, _summary, _step, alpha, ty, py = sess.run([cost, accuracy, validate_summary_op, global_step, self.alpha, true_y, pred_y],
                                                            feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                    if flag:
                        summary = _summary
                        step = _step
                        flag = False
                        alpha = alpha
                        ty = ty
                        py = py
                print 'all samples={}, correct prediction={}'.format(cnt, acc)
                test_summary_writer.add_summary(summary, step)
                saver.save(sess, save_dir, global_step=step)
                print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, loss / cnt, acc / cnt)
                if acc / cnt > max_acc:
                    max_acc = acc / cnt
                    max_alpha = alpha
                    max_ty = ty
                    max_py = py

            print 'Optimization Finished! Max acc={}'.format(max_acc)
            fp = open('weight.txt', 'w')
            for y1, y2, ws in zip(max_ty, max_py, max_alpha):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws]) + '\n')

            print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
                self.learning_rate,
                self.n_iter,
                self.batch_size,
                self.n_hidden,
                self.l2_reg
            )

    def get_batch_data(self, x, sen_len, y, target_words, batch_size, keep_prob1, keep_prob2, is_shuffle=True):
        for index in batch_index(len(y), batch_size, 1, is_shuffle):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.aspect_id: target_words[index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
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
        type_=FLAGS.method
    )
    lstm.run()


if __name__ == '__main__':
    tf.app.run()
