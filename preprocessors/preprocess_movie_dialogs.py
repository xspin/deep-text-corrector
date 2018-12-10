"""Preprocesses Cornell Movie Dialog data."""
import nltk
import tensorflow as tf
import random

tf.app.flags.DEFINE_string("raw_data", "../../dataset/cornell movie-dialogs corpus/movie_lines.txt", "Raw data path")
tf.app.flags.DEFINE_string("out_file", "../movie_lines.txt", "File to write preprocessed data")
tf.app.flags.DEFINE_string("val_file", "../val.txt", "File to write val data")
tf.app.flags.DEFINE_string("test_file", "../test.txt", "File to write test data")

FLAGS = tf.app.flags.FLAGS

def choose():
    r = random.random()
    if r > 0.2: return 0 # train
    if r < 0.2/3: return 2 # val
    return 1 # test

def main(_):
    with open(FLAGS.raw_data, "rb") as raw_data, \
            open(FLAGS.test_file, "w") as testout, \
            open(FLAGS.val_file, "w") as valout, \
            open(FLAGS.out_file, "w") as trainout:
        for line in raw_data:
            parts = line.split(b" +++$+++ ")
            dialog_line = parts[-1]
            s = dialog_line.strip().lower().decode("utf-8", "ignore")
            preprocessed_line = " ".join(nltk.word_tokenize(s))
            out = None
            k = choose()
            if k==0: out = trainout
            elif k==1: out = testout
            else: out = valout
            out.write(preprocessed_line + "\n")

if __name__ == "__main__":
    tf.app.run()
    print('done')
