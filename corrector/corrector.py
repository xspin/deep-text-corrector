import nltk.tokenize as tok
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer as detok
import tensorflow as tf

from .correct_text import train, decode, decode_sentence, evaluate_accuracy, create_model,\
    get_corrective_tokens, DefaultPTBConfig, DefaultMovieDialogConfig, get_reader

from .text_corrector_data_readers import PTBDataReader, MovieDialogReader

def find_error(src_tokens, tgt_tokens):
    src = nltk.pos_tag(src_tokens)
    tgt = nltk.pos_tag(tgt_tokens)
    src.append('EOS')
    tgt.append('EOS')
    record = []
    i, j = 0, 0
    while src[i]!='EOS' and tgt[j]!='EOS':
        if src[i][0]==tgt[j][0]: 
            pass
        elif src[i+1][0]==tgt[j][0]: # a word is removed
            record.append([src[i], None])
            i += 1
        elif src[i][0]==tgt[j+1][0]: # a word is added
            record.append([None, tgt[j]])
            j += 1
        else:
            record.append([src[i], tgt[j]])
        i += 1
        j += 1
    while src[i]!='EOS':
        record.append([src[i], None])
        i += 1
    while tgt[j]!='EOS':
        record.append([None, tgt[j]])
        j += 1

            
    def get_error_type(a, b):
        ta, tb = None, None
        if a: ta = ErrorType[a[1]] if a[1] in ErrorType else '#'
        if b: tb = ErrorType[b[1]] if b[1] in ErrorType else '#'
        if not tb: return ta
        return tb

    error_type = [get_error_type(*r) for r in record]
    errors = [ErrorName[t] for t in error_type]
    return record, errors

# ErrorName = {
#     'PREP': '介词',
#     'NOUN': '名词',
#     'PRONOUN': '代词',
#     'ART': '冠词',
#     'VERB': '动词',
#     'PRED': '谓词',
#     'ADJ': '形容词',
#     'ADV': '副词',
#     'PUNC': '标点',
#     '#': '未知',
# }
ErrorName = {
    'PREP': 'Preposition', #介词
    'NOUN': 'Noun', #名词
    'PRONOUN': 'Pronoun', #代词
    'ART': 'Article', #冠词',
    'VERB': 'Verb', #'动词',
    'PRED': 'Predicate', #'谓词',
    'ADJ': 'Adjective', #'形容词',
    'ADV': 'Adverb', #'副词',
    'PUNC': 'Punctuation', #'标点',
    '#': 'Unknown' #'未知',
}
ErrorType = {
    'DT': 'ART',
    'WDT': 'ART',
    'IN': 'PREP',
    'JJ': 'ADJ',
    'JJR': 'ADJ',
    'JJS': 'ADJ',
    'RB': 'ADV',
    'RBR': 'ADV',
    'RBS': 'ADV',
    'NN': 'NOUN',
    'NNS': 'NOUN',
    'NNP': 'NOUN',
    'NNPS': 'NOUN',
    'TO': 'IN',
    'VB': 'VERB',
    'VBG': 'VERB',
    'VBN': 'VERB',
    'VBP': 'VERB',
    'VBZ': 'VERB',
    'WP': 'PRONOUN',
    'WP$': 'PRONOUN',
    'PRP': 'PRONOUN',
    'PRP$': 'PRONOUN',
    '.': 'PUNC',
    ',': 'PUNC',
    '?': 'PUNC',
    '!': 'PUNC',
    '"': 'PUNC',
    "'": 'PUNC',
    ':': 'PUNC',
}

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

    def correct(self, sentence):
        sentence = sentence.lower()
        tokens = tok.word_tokenize(sentence)
        result = decode_sentence(self.sess, self.model, self.data_reader, tokens, 
                                    corrective_tokens=self.corrective_tokens, verbose=False)

        alteration, error = find_error(tokens, result)
        result = detok().detokenize(result)
        return result, alteration, error