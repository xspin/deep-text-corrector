from corrector import Corrector

reader = 'Conll'

if reader == 'MovieDialog':
    train_path = 'dataset/movie_lines.txt'
    test_path = 'dataset/test.txt'
    model_path = 'dialog_correcter_model'
elif reader == 'Conll':
    train_path = 'dataset/CONLL/train.txt'
    test_path = 'dataset/CONLL/test.txt'
    model_path = 'conll_correcter_model'


corrector = Corrector(train_path, test_path, model_path, reader=reader)
corrector.train()