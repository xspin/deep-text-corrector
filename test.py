
import os
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict

from correct_text import train, decode, decode_sentence, evaluate_accuracy, create_model,\
    get_corrective_tokens, DefaultPTBConfig, DefaultMovieDialogConfig, get_reader

from text_corrector_data_readers import PTBDataReader, MovieDialogReader
import nltk.tokenize as tok

# data_reader = MovieDialogReader(config, train_path, dropout_prob=0.25, replacement_prob=0.25, dataset_copies=1)
def test(fname, N=20):
    i = 0
    sents = list(data_reader.read_tokens(fname))
    decodings = decode(sess, model=model, data_reader=data_reader,
                        data_to_decode=sents, verbose=False)
    # Write the decoded tokens to stdout.
    for sent, tokens in zip(sents, decodings):
        print(" Input: {}".format(' '.join(sent)))
        print("Output: {}".format(' '.join(tokens)))
        i += 1
        if N and i>N: break

def loop():
    while True:
        sent = str(input('\n Input: '))
        if sent in ['q', 'quit', 'exit']: break
        # sent = tok.word_tokenize(sent)
        # sent = ' '.join(sent)
        decoded = decode_sentence(sess, model, data_reader, sent, corrective_tokens=corrective_tokens, verbose=False)
        print('Output:', decoded)

train_path = 'movie_lines.txt'
test_path = 'test.txt'
example_path = 'example.txt'

config = DefaultMovieDialogConfig()
data_reader = get_reader(config, train_path, test_path)
corrective_tokens = get_corrective_tokens(data_reader, train_path)

with tf.Session() as sess:
    model, _ = create_model(sess, True, config=config)
    test(example_path, None)
    loop()
print('exit')