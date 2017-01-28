import sys
import os
import numpy as np
import matplotlib.pyplot as plt
DIR = './Results'

def lectura (filename):
    values = []
    f = open(DIR+'/'+filename)
    for i,line in enumerate(f):
        values.append(line)
    return values

#f = open('./Results')
tr_acc=[]
tr_loss=[]
val_acc=[]
val_loss=[]

#With this loop we want to create 4 sorted lists in order to print the plots given a batch size and an optimization.
#Each matching position of the sorted lists corresponds to the same execution.
for name in sorted(os.listdir(DIR)):
    #noms_fitxers.append(i)
    name=name.split('_', 4)
    batches=name[2]
    if name[0]=='tr':
        if name[1]=='acc':
            s='_'
            name=s.join(name)
            tr_acc.append(name)
        elif name[1]=='loss' or name[1]=='losses':
            s='_'
            name=s.join(name)
            tr_loss.append(name)
    elif name[0]== 'val':
        if name[1]=='acc':
            s='_'
            name=s.join(name)
            val_acc.append(name)
        elif name[1]=='loss' or name[1]=='losses':
            s='_'
            name=s.join(name)
            val_loss.append(name)

acc1=sorted(tr_acc)
acc2=sorted(val_acc)
loss1=sorted(tr_loss)
loss2=sorted(val_loss)

for i in range(len(tr_acc)):
    name=tr_acc[i].split('_')

    fig=plt.figure(i)

    fig.suptitle('Number of batches: ' + name[2] + ' Optimizer: ' + name[3])
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    ax1.plot(lectura(acc1[i]),'r',lectura(acc2[i]),'g')
    ax1.set_xlabel('Number of epochs')
    ax1.set_ylabel('Accuracy (%)')

    ax2.plot(lectura(loss1[i]),'r',lectura(loss2[i]),'g')
    ax2.set_xlabel('Number of epochs')
    ax2.set_ylabel('Loss (%)')

    plt.show()
