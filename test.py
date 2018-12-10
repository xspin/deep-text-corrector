
import os
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict

from correct_text import train, decode, decode_sentence, evaluate_accuracy, create_model,\
    get_corrective_tokens, DefaultPTBConfig, DefaultMovieDialogConfig, get_reader

from text_corrector_data_readers import PTBDataReader, MovieDialogReader

# data_reader = MovieDialogReader(config, train_path, dropout_prob=0.25, replacement_prob=0.25, dataset_copies=1)
def test():
    with tf.Session() as sess:
        model, _ = create_model(sess, True, config=config)
        sents = list(data_reader.read_tokens('example.txt'))
        decodings = decode(sess, model=model, data_reader=data_reader,
                            data_to_decode=sents, verbose=False)
        # Write the decoded tokens to stdout.
        for sent, tokens in zip(sents, decodings):
            print("Input: {}".format(' '.join(sent)))
            print("Output: {}".format(' '.join(tokens)))

train_path = 'movie_lines.txt'
test_path = 'test.txt'

config = DefaultMovieDialogConfig()
data_reader = get_reader(config, train_path, test_path)
corrective_tokens = get_corrective_tokens(data_reader, train_path)
# train(data_reader)
# with tf.Session() as sess:
#     model, _ = create_model(sess, True, config=config)
#     decoded = decode_sentence(sess, model, data_reader, "you must have girlfriend", corrective_tokens=corrective_tokens)
test()