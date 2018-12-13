import nltk.tokenize as tok
import tensorflow as tf

from .correct_text import train, decode, decode_sentence, evaluate_accuracy, create_model,\
    get_corrective_tokens, DefaultPTBConfig, DefaultMovieDialogConfig, get_reader

from corrector.text_corrector_data_readers import PTBDataReader, MovieDialogReader


class Corrector:
    def __init__(self, train_path, test_path, model_path, reader='MovieDialog'):
        self.config = DefaultMovieDialogConfig()
        self.config.model_path = model_path
        # self.data_reader = get_reader(self.config, train_path, test_path, reader=reader, process=False)
        self.train_path = train_path
        self.test_path = test_path
        self.reader = reader

    def train(self):
        data_reader = get_reader(self.config, self.train_path, self.test_path, reader=self.reader, process=True)
        train(data_reader)

    def corrector_init(self):
        self.sess = tf.Session()
        self.data_reader = get_reader(self.config, self.train_path, self.test_path, reader=self.reader, process=False)
        self.corrective_tokens = get_corrective_tokens(self.data_reader, self.train_path)
        self.model, _ = create_model(self.sess, True, config=self.config)

    def correct(self, sentence, verbose=False):
        result = decode_sentence(self.sess, self.model, self.data_reader, sentence, 
                                    corrective_tokens=self.corrective_tokens, verbose=verbose)
        return result