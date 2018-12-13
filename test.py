from corrector import Corrector

train_path = 'dataset/movie_lines.txt'
test_path = 'dataset/test.txt'
example_path = 'dataset/example.txt'
model_path = 'dialog_correcter_model'


corrector = Corrector(train_path, test_path, model_path)
corrector.corrector_init()
while True:
    sent = str(input("\n Input: "))
    if sent in ['q', 'exit', 'quit']: break
    result = corrector.correct(sent)
    print('Output:', result)