# coding: utf-8

# In[1]:

from __future__ import print_function

import pickle
import sys

import numpy
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import lasagne
import time

from custom_layers import CustomEmbedding, MatchLSTM, FakeFeatureDot2Layer


# In[2]:

def prepare(df):
    seqs_premise = []
    seqs_hypothesis = []
    for cc in df['sentence1']:
        seqs_premise.append(cc)
    for cc in df['sentence2']:
        seqs_hypothesis.append(cc)
    seqs_p = seqs_premise
    seqs_h = seqs_hypothesis

    lengths_p = [len(s) for s in seqs_p]
    lengths_h = [len(s) for s in seqs_h]

    n_samples = len(seqs_p)
    maxlen_p = numpy.max(lengths_p) + 1
    maxlen_h = numpy.max(lengths_h) + 1

    premise = numpy.zeros((n_samples, maxlen_p))
    hypothesis = numpy.zeros((n_samples, maxlen_h))
    premise_masks = numpy.zeros((n_samples, maxlen_p))
    hypothesis_masks = numpy.zeros((n_samples, maxlen_h))
    for idx, [s_t, s_h] in enumerate(zip(seqs_p, seqs_h)):
        assert lengths_h[idx] == len(s_h)
        premise[idx, :lengths_p[idx]] = s_t
        premise_masks[idx, :lengths_p[idx]] = 1
        hypothesis[idx, :lengths_h[idx]] = s_h
        hypothesis_masks[idx, :lengths_h[idx]] = 1
    labels = []
    for gl in df['gold_label']:
        if gl == 'entailment':
            value = 2
        elif gl == 'contradiction':
            value = 1
        elif gl == 'neutral':
            value = 0
        else:
            raise ValueError('unknown gold_label {0}'.format(gl))
        labels.append(value)

    labels = np.array(labels)
    return (premise.astype('int32'),
            premise_masks.astype('int32'),
            hypothesis.astype('int32'),
            hypothesis_masks.astype('int32'),
            labels.astype('int32'))


# In[3]:

print('Loading data ...')
dev_df, test_df = (None, None)
with open('./snli/converted_dev.pkl', 'rb') as f:
    print('Loading dev ...')
    dev_df = pickle.load(f)
    print(len(dev_df))
    filtered_s2 = dev_df.sentence2.apply(lambda s2: len(s2) != 0)
    dev_df = dev_df[filtered_s2]
    print(len(dev_df))
    dev_df = dev_df[dev_df.gold_label != '-']
    dev_df = dev_df.reset_index()
    print(len(dev_df))
with open('./snli/converted_test.pkl', 'rb') as f:
    print('Loading test ...')
    test_df = pickle.load(f)
    print(len(test_df))
    filtered_s2 = test_df.sentence2.apply(lambda s2: len(s2) != 0)
    test_df = test_df[filtered_s2]
    print(len(test_df))
    test_df = test_df[test_df.gold_label != '-']
    test_df = test_df.reset_index()
    print(len(test_df))
# In[7]:

premise_max = 82 + 1
hypothesis_max = 62 + 1


# In[8]:

def main(k=300, batch_size=30,
         display_freq=100,
         save_freq=1000,
         load_previous=False):
    print('k: {}'.format(k))
    print('batch_size: {}'.format(batch_size))
    print('display_frequency: {}'.format(display_freq))
    print('save_frequency: {}'.format(save_freq))
    print('load previous: {}'.format(load_previous))
    save_filename = '/tmp/mlstm_model.npz'
    print("Building network ...")
    premise_var = T.imatrix('premise_var')
    premise_mask = T.imatrix('premise_mask')
    hypo_var = T.imatrix('hypo_var')
    hypo_mask = T.imatrix('hypo_mask')
    unchanged_W = pickle.load(open('./snli/unchanged_W.pkl', 'rb'))
    unchanged_W = unchanged_W.astype('float32')
    unchanged_W_shape = unchanged_W.shape
    oov_in_train_W = pickle.load(open('./snli/oov_in_train_W.pkl', 'rb'))
    oov_in_train_W = oov_in_train_W.astype('float32')
    oov_in_train_W_shape = oov_in_train_W.shape
    print('unchanged_W.shape: {0}'.format(unchanged_W_shape))
    print('oov_in_train_W.shape: {0}'.format(oov_in_train_W_shape))
    # best hypoparameters
    p = 0.3
    learning_rate = 0.001
    # learning_rate = theano.shared(0.001)
    # learning_rate = 0.003
    # l2_weight = 0.0003
    l2_weight = 0.

    l_premise = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_var)
    l_premise_mask = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_mask)
    l_hypo = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_var)
    l_hypo_mask = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_mask)

    premise_embedding = CustomEmbedding(l_premise, unchanged_W, unchanged_W_shape,
                                        oov_in_train_W, oov_in_train_W_shape,
                                        p=p)
    # weights shared with premise_embedding
    hypo_embedding = CustomEmbedding(l_hypo, unchanged_W=premise_embedding.unchanged_W,
                                     unchanged_W_shape=unchanged_W_shape,
                                     oov_in_train_W=premise_embedding.oov_in_train_W,
                                     oov_in_train_W_shape=oov_in_train_W_shape,
                                     p=p,
                                     dropout_mask=premise_embedding.dropout_mask)
    hypo_embedding = FakeFeatureDot2Layer(hypo_embedding)
    mlstm = MatchLSTM(hypo_embedding, k, peepholes=False, mask_input=l_hypo_mask,
                      encoder_input=premise_embedding, encoder_mask_input=l_premise_mask,
                      )
    p = 0.
    if p > 0.:
        print('apply dropout rate {} to decoder'.format(p))
        mlstm = lasagne.layers.DropoutLayer(mlstm, p)
    l_softmax = lasagne.layers.DenseLayer(
            mlstm, num_units=3,
            nonlinearity=lasagne.nonlinearities.softmax)
    if load_previous:
        print('loading previous saved model ...')
        # And load them again later on like this:
        with np.load(save_filename) as f:
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
    val_fn = theano.function([premise_var, premise_mask, hypo_var, hypo_mask, target_var],
                             [test_loss, test_acc])
    predict_fn = theano.function([premise_var, premise_mask, hypo_var, hypo_mask],
                                 test_predict_cls)
    print("Testing ...")
    print('dev_df.shape: {0}'.format(dev_df.shape))
    print('test_df.shape: {0}'.format(test_df.shape))
    try:
        print("Starting evaluating...")
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for start_i in range(0, len(dev_df), batch_size):
            batched_df = dev_df[start_i:start_i + batch_size]
            ps, p_masks, hs, h_masks, labels = prepare(batched_df)
            err, acc = val_fn(ps, p_masks, hs, h_masks, labels)
            val_err += err
            val_acc += acc
            val_batches += 1
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0

        for start_i in range(0, len(test_df), batch_size):
            batched_df = test_df[start_i:start_i + batch_size]
            ps, p_masks, hs, h_masks, labels = prepare(batched_df)
            err, acc = val_fn(ps, p_masks, hs, h_masks, labels)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

        # Optionally, you could now dump the network weights to a file like this:
        # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)
    except KeyboardInterrupt:
        print('exit ...')


# In[9]:

if __name__ == '__main__':

    main(batch_size=30, load_previous=True)

