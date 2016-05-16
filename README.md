reasoning_attention
===================
Unofficial implementation algorithm of paper "Reasoning About Entailment With Neural Attention".

Based on Lasagne.

Paper see [http://arxiv.org/abs/1509.06664][1].

Requirements
===========
* Python 3
* Lasagne

Run
===
At source root dir

First extracts preprocessed SNLI data
`./extract_data.sh`

Then run:
`python3 ./snli_reasoning_attention.py [condition|attention|word_by_word]`


[1]: http://arxiv.org/abs/1509.06664
