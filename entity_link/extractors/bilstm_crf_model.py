#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-24 14:23
# @Author  : zhangzhen
# @Site    : 
# @File    : bilstm_crf_model.py.py
# @Software: PyCharm
from typing import Any
import tensorflow as tf
import logging
import time
import sys
import os

from entity_link.extractors.utils import batch_yield, pad_sequences, conlleval

logger = logging.getLogger(__name__)


class BiLstmCRF(object):

    def __init__(self, config, embeddings, tag2label, vocab, eval_perl, shuffle=True, use_crf=True,
                 update_embedding=True, batch_size=64, epoch=10, hidden_dim=60, keep_prob=0.5, optimizer='Adam',
                 lr=0.001, clip_grad=5.0, **kwargs: Any):
        self.config = config
        self.embeddings = embeddings

        self.batch_size = batch_size
        self.epoch_num = epoch
        self.hidden_dim = hidden_dim
        self.use_crf = use_crf
        self.update_embedding = update_embedding
        self.dropout_keep_prob = keep_prob
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = shuffle
        self.eval_perl = eval_perl

        self.model_path = kwargs['model_path']
        self.summary_path = kwargs['summary_path']
        self.result_path = kwargs['result_path']

        self.makedirs(self.model_path)
        self.makedirs(self.summary_path)
        self.makedirs(self.result_path)

    def makedirs(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def build_graph(self):
        self.__add_placeholders()
        self.__lookup_layer_op()
        self.__bi_lstm_layer_op()
        if not self.use_crf:
            self.__soft_max_pred_op()
        self.__loss_op()
        self.__train_step_op()
        self.__init_op()

    def __add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def __lookup_layer_op(self):
        with tf.variable_scope("embedding"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def __bi_lstm_layer_op(self):
        with tf.variable_scope("bilstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                                inputs=self.word_embeddings,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("projection"):
            W = tf.get_variable(name="W", shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

            b = tf.get_variable(name="b", shape=[self.num_tags], initializer=tf.zeros_initializer(), dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def __soft_max_pred_op(self):
        self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
        self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def __loss_op(self):
        if self.use_crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                       tag_indices=self.labels,
                                                                                       sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def __train_step_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def __init_op(self):
        self.init_op = tf.global_variables_initializer()

    def __add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.__add_summary(sess)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids, self.sequence_lengths: seq_len_list}

        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_

        if lr is not None:
            feed_dict[self.lr_pl] = lr

        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def run_one_epoch(self, sess, train, dev, epoch, saver):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')

            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)

            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)

            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                logging.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path + os.sep + 'checkpoints', global_step=step_num)

        logging.info('======================validation / test======================')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)

        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def dev_one_epoch(self, sess, dev):
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):

        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.use_crf:
            logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch + 1) if epoch is not None else 'test'

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path, eval_perl=self.eval_perl):
            logging.info(_)

    def run_one_case(self, sess, sent):
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag
