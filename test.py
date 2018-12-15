from corrector import Corrector

train_path = 'dataset/movie_lines.txt'
test_path = 'dataset/test.txt'
example_path = 'dataset/example.txt'
model_path = 'dialog_correcter_model'


corrector = Corrector(train_path, test_path, model_path)
corrector.corrector_init()

def process(sent):
    result = corrector.correct(sent)
    result, alt, err = corrector.correct(sent)
    print('Output:', result)
    print('-'*30)
    for r, e in zip(alt, err):
        x = r[0][0] if r[0] else None
        y = r[1][0] if r[1] else None
        if not x:
            print('{:>13} ERROR      Add: {} '.format(e, y))
        elif not y:
            print('{:>13} ERROR   Remove: {}  '.format(e, x))
        else:
            print('{:>13} ERROR  Replace: {} -> {}'.format(e, x, y))
    print('-'*30)

with open(example_path, 'r') as f:
    for sentence in f:
        sentence = sentence.strip()
        print(' input:', sentence)
        process(sentence)


while True:
    sent = str(input("\n Input: "))
    if sent in ['q', 'exit', 'quit']: break
    process(sent)