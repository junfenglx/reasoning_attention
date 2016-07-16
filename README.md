Attention on SNLI
===================
Unofficial implementation algorithms of attention models on SNLI dataset.

Current include papers:

1. "Reasoning About Entailment With Neural Attention", [arXiv:1509.06664][1]
2. "Learning Natural Language Inference with LSTM", [arXiv:1512.08849][2]

Based on Lasagne.


Requirements
===========
* Python 3
* Lasagne
* some others(best install Anaconda)

Run
===
At source root dir

First extracts preprocessed SNLI data
`./extract_data.sh`

Then run:
`python3 ./snli_reasoning_attention.py [condition|attention|word_by_word]`


Or run:
`python3 ./snli_match_lstm.py`


Results
======

The learning curve of word by word attention:
![wordbyword_attention](./wordbyword_attention_dropout0.2.png)

The learning curve of match LSTM with word embedding:
![mlstm_word2vec_embedding](./mlstm_word2vec_embedding.png)

Notes
=====
About word by word attention:

1. The test acc of match LSTM is 1% smaller than the original paper, 82.31%


About match LSTM:

1. The test acc of match LSTM is 1% smaller than the original paper.
2. not decay learning rate when train
3. using Word2Vec pre-trained model
4. tune OOV words in train dataset, like in `Reasoning Attention`



[1]: http://arxiv.org/abs/1509.06664
[2]: http://arxiv.org/abs/1512.08849
[3]: 
