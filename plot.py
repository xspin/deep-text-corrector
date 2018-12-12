import matplotlib.pyplot as plt

log_path = 'dialog_correcter_model/log.txt'

step = []
loss = []
avgloss = []
print('Processing data ...')
with open(log_path, 'r') as fhd:
    for line in fhd:
        line = line[24:].strip()
        if line[:4].lower() == 'step':
            cols = line.split(', ')
            k = cols[0].find('/')
            step.append(int(cols[0][5:k]))
            loss.append(float(cols[3][5:]))
            avgloss.append(float(cols[4][8:]))



print('Ploting ...')
plt.figure()
plt.subplot(211)
plt.plot(step, loss)
plt.title('Batch Loss')
# plt.xticks()
plt.xlim(min(step), max(step))

plt.subplot(212)
plt.plot(step, avgloss)
plt.title('Averaged Loss')
plt.xlabel('Step')
plt.ylim(0,3)
plt.xlim(min(step), max(step))

plt.show()

plt.savefig('loss.png')