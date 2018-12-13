from corrector import Corrector

train_path = 'dataset/movie_lines.txt'
test_path = 'dataset/test.txt'
example_path = 'dataset/example.txt'
model_path = 'dialog_correcter_model'


corrector = Corrector(train_path, test_path, model_path)
corrector.train()