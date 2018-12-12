import nltk
import nltk.tokenize as tok
tokens = tok.word_tokenize('this is what i want to say.')
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
print(nltk.pos_tag(['left']))
