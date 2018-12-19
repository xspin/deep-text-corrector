import matplotlib.pyplot as plt

log_path = 'conll_correcter_model/log.txt'

step = []
loss = []
avloss = []
exploss = []
print('Processing data ...')
with open(log_path, 'r') as fhd:
    for line in fhd:
        line = line[24:].strip()
        if line[:5].lower() == 'step ':
            cols = line.split(', ')
            k = cols[0].find('/')
            step.append(int(cols[0][5:k]))
            loss.append(float(cols[3][5:]))
            avloss.append(float(cols[4][8:]))
            exploss.append(float(cols[4][9:]))



print('Ploting ...')
plt.figure(figsize=(12,7))
plt.subplot(311)
plt.plot(step, loss)
# plt.title('Batch Loss')
# plt.xticks()
plt.xlim(min(step), max(step))
plt.ylabel('Step Loss')

plt.subplot(312)
plt.plot(step, avloss)
# plt.title('Averaged Loss')
plt.xlabel('Step')
plt.ylabel('Av Loss')
# plt.ylim(0,3)
plt.xlim(min(step), max(step))

plt.subplot(313)
plt.plot(step, exploss)
# plt.title('Exp Loss')
plt.xlabel('Step')
plt.ylabel('Exp Loss')
# plt.ylim(0,3)
plt.xlim(min(step), max(step))

plt.savefig('loss_conll.png')
plt.show()