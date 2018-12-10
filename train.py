from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
# import pandas as pd
from collections import defaultdict

# from sklearn.metrics import roc_auc_score, accuracy_score
# import nltk

from correct_text import train, decode, decode_sentence, evaluate_accuracy, create_model,\
    get_corrective_tokens, DefaultPTBConfig, DefaultMovieDialogConfig, get_reader

from text_corrector_data_readers import PTBDataReader, MovieDialogReader


# os.environ['CUDA_VISIBLE_DEVICES'] = ''


train_path = 'movie_lines.txt'
test_path = 'test.txt'

config = DefaultMovieDialogConfig()
data_reader = get_reader(config, train_path, test_path)
train(data_reader)