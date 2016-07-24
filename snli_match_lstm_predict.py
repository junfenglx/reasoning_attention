# -*- coding: utf-8 -*-

# Created by junfeng on 7/16/16.

# logging config

import logging
import pickle
import sys
import time

import numpy
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import lasagne
from nltk import TreebankWordTokenizer

from custom_layers import CustomEmbedding, MatchLSTM, FakeFeatureDot2Layer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
word_tokenize = TreebankWordTokenizer().tokenize


class MatchLSTMModel(object):
    LABEL2NAME = {0: 'neutral',
                  1: 'contradiction',
                  2: 'entailment'}

    def __init__(self,
                 dict_file='./snli/dictionary.pkl',
                 model_file='./snli/mlstm_model.npz',
                 unchanged_W_file='./snli/unchanged_W.pkl',
                 oov_in_train_file='./snli/oov_in_train_W.pkl',
                 k=300, p=0.3):
        self.dict_file = dict_file
        self.model_file = model_file
        self.unchanged_W_file = unchanged_W_file
        self.oov_in_train_file = oov_in_train_file
        self.k = k
        self.p = p

        self.dictionary = None
        self.predict_fn = None

        self.load_data()
        self.build_model()

    def load_data(self):
        with open(self.dict_file, 'rb') as f:
            self.dictionary = pickle.load(f)

    def word2id(self, sentence):
        sent = []
        words = word_tokenize(sentence)

        for w in words:
            if w in self.dictionary:
                sent.append(self.dictionary[w])
            else:
                print('Warning: {} not in dictionary'.format(w))
        return sent

    def prepare(self, samples):
        seqs_premise = []
        seqs_hypothesis = []
        for i, (p, h) in enumerate(samples, start=1):
            p_ids = self.word2id(p)
            h_ids = self.word2id(h)
            if not (p_ids and h_ids):
                print('sample {} has empty sentence, skiped'.format(i))
                continue
            seqs_premise.append(p_ids)
            seqs_hypothesis.append(h_ids)
        seqs_p = seqs_premise
        seqs_h = seqs_hypothesis

        lengths_p = [len(s) for s in seqs_p]
        lengths_h = [len(s) for s in seqs_h]

        n_samples = len(seqs_p)
        maxlen_p = numpy.max(lengths_p) + 1
        maxlen_h = numpy.max(lengths_h) + 1

        premise = numpy.zeros((n_samples, maxlen_p), dtype='int32')
        hypothesis = numpy.zeros((n_samples, maxlen_h), dtype='int32')
        premise_masks = numpy.zeros((n_samples, maxlen_p), dtype='int32')
        hypothesis_masks = numpy.zeros((n_samples, maxlen_h), dtype='int32')
        for idx, [s_t, s_h] in enumerate(zip(seqs_p, seqs_h)):
            assert lengths_h[idx] == len(s_h)
            premise[idx, :lengths_p[idx]] = s_t
            premise_masks[idx, :lengths_p[idx]] = 1
            hypothesis[idx, :lengths_h[idx]] = s_h
            hypothesis_masks[idx, :lengths_h[idx]] = 1

        return (premise, premise_masks,
                hypothesis, hypothesis_masks)

    def build_model(self):
        premise_max = 82 + 1
        hypothesis_max = 62 + 1

        print("Building network ...")
        premise_var = T.imatrix('premise_var')
        premise_mask = T.imatrix('premise_mask')
        hypo_var = T.imatrix('hypo_var')
        hypo_mask = T.imatrix('hypo_mask')
        unchanged_W = pickle.load(open(self.unchanged_W_file, 'rb'))
        unchanged_W = unchanged_W.astype('float32')
        unchanged_W_shape = unchanged_W.shape
        oov_in_train_W = pickle.load(open(self.oov_in_train_file, 'rb'))
        oov_in_train_W = oov_in_train_W.astype('float32')
        oov_in_train_W_shape = oov_in_train_W.shape
        print('unchanged_W.shape: {0}'.format(unchanged_W_shape))
        print('oov_in_train_W.shape: {0}'.format(oov_in_train_W_shape))

        l_premise = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_var)
        l_premise_mask = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_mask)
        l_hypo = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_var)
        l_hypo_mask = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_mask)

        premise_embedding = CustomEmbedding(l_premise, unchanged_W, unchanged_W_shape,
                                            oov_in_train_W, oov_in_train_W_shape,
                                            p=self.p)
        # weights shared with premise_embedding
        hypo_embedding = CustomEmbedding(l_hypo, unchanged_W=premise_embedding.unchanged_W,
                                         unchanged_W_shape=unchanged_W_shape,
                                         oov_in_train_W=premise_embedding.oov_in_train_W,
                                         oov_in_train_W_shape=oov_in_train_W_shape,
                                         p=self.p,
                                         dropout_mask=premise_embedding.dropout_mask)
        hypo_embedding = FakeFeatureDot2Layer(hypo_embedding)
        mlstm = MatchLSTM(hypo_embedding, self.k, peepholes=False, mask_input=l_hypo_mask,
                          encoder_input=premise_embedding, encoder_mask_input=l_premise_mask,
                          )
        self.p = 0.
        if self.p > 0.:
            print('apply dropout rate {} to decoder'.format(p))
            mlstm = lasagne.layers.DropoutLayer(mlstm, p)
        l_softmax = lasagne.layers.DenseLayer(
                mlstm, num_units=3,
                nonlinearity=lasagne.nonlinearities.softmax)
        print('loading pre-trained model ...')
        # And load them again later on like this:
        with np.load(self.model_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(l_softmax, param_values)

        target_var = T.ivector('target_var')

        test_prediction = lasagne.layers.get_output(l_softmax, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # lasagne.objectives.categorical_accuracy()
        # As a bonus, also create an expression for the classification accuracy:
        test_predict_cls = T.argmax(test_prediction, axis=1)
        test_acc = T.mean(T.eq(test_predict_cls, target_var),
                          dtype=theano.config.floatX)

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        self.predict_fn = theano.function([premise_var, premise_mask, hypo_var, hypo_mask],
                                     test_predict_cls)

    def train(self):
        pass

    def predict(self, samples):
        """
        prediction function
        :param samples: list of tuple(premise, hypothesis)
        :return:
        """
        (premise, premise_masks, hypothesis, hypothesis_masks) = self.prepare(samples)
        predicted_class = self.predict_fn(premise, premise_masks, hypothesis, hypothesis_masks)
        for name in map(lambda l: self.LABEL2NAME[l], predicted_class):
            print(name)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: {} sample_file'.format(sys.argv[0]))
        sys.exit(0)
    sample_file = sys.argv[1]
    samples = []
    with open(sample_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            p = lines[i]
            h = lines[i+1]
            print((p, h))
            samples.append((p, h))

    model = MatchLSTMModel()
    model.predict(samples)
