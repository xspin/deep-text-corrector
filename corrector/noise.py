import re
import numpy as np

VOCAB = {}
VOCAB['prep'] = ['in', 'at', 'on', 'of', 'behind', 'via', 'under', 'for', 'beyond', 'towards', 'since', 
    'beside', 'inside', 'outside', 'before', 'above', 'with', 'from', 'between', 'upon', 'into', 
    'after', 'as', 'by', 'about', 'without']
VOCAB['art'] = ['the', 'a', 'an']
VOCAB['pred'] = ['am', 'is', 'are', 'was', 'were', 'have', "'s", "'ve", 'had', 'has', "'m", 'do', 'does']
VOCAB['punc'] = ['.', ',', ';', ':', '"', "'", '?', '!']

keep_prob = 0.6
drop_prob = 0.2

def get_type(word):
    tp = []
    idx = []
    verb_regex = r'[a-z]+(ing|ed|en)$'
    t = None
    if word in VOCAB['prep']: t = 'prep'
    elif word in VOCAB['art']: t = 'art'
    elif word in VOCAB['pred']: t = 'pred'
    elif word in VOCAB['punc']: t = 'punc'
    elif re.match(verb_regex, word): t = 'verb'
    return t

def add_noise(token):
    '''add noise to the sentence randomly choosing the following 6 erroneous types
        - preposition
        - article
        - predicate
        - punctuation
        - noun
        - verb
    '''
    # types = ['prep', 'noun', 'article', 'verb', 'pred', 'punc']
    # No change with propability 0.1
    if np.random.rand()<keep_prob: return token
    tp = get_type(token)
    if tp in ['prep', 'art', 'pred', 'punc']:
        if np.random.rand() < drop_prob: token = ''
        else: token = np.random.choice(VOCAB[tp])
    elif tp == 'verb':
        if token[-3:] == 'ing': token = token[:-3]+'ed'
        elif token[-2:] == 'ed': 
            if np.random.rand() < 0.5: token = token[:-2]+'en'
            else: token = token[:-2]+'ing'
        elif token[-2:] == 'en':  token = token[:-2]+'ed'
    else:
        pass
        #TODO: for nouns and other types
    return token

if __name__ == '__main__':
    tokens = ['i', 'hated', 'am', 'in', 'the', 'road', 'of', 'there', 'doing', 'haven', '!']
    for token in tokens:
        print(add_noise(token))