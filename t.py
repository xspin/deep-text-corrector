import nltk
import nltk.tokenize as tok
from corrector.corrector import find_error

def process(sent, target):
    tokens = tok.word_tokenize(sent)
    result = tok.word_tokenize(target)
    alt, err = find_error(tokens, result)
    print(' Input:', sent)
    print('Output:', target)
    print(alt)
    for r, e in zip(alt, err):
        x = r[0][0] if r[0] else None
        y = r[1][0] if r[1] else None
        if not x:
            print('{:>13} ERROR,     Add: {} '.format(e, y))
        elif not y:
            print('{:>13} ERROR,  Remove: {}  '.format(e, x))
        else:
            print('{:>13} ERROR, Replace: {} -> {}'.format(e, x, y))
    print('')

sent = "what you fuck the doing"
target = "what are you fucking doing?"

sent = 'too young too simple'
target = 'too young, too simple.'

sent = 'what fuck'
target = 'what the fuck!'
process(sent, target)