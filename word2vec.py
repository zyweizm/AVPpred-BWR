from Bio import SeqIO
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
import os
from tools import read_fasta,supple_X
import gensim
import gzip
import os
import glob
import csv
import multiprocessing
Path('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/data/train').mkdir(exist_ok=True, parents=True)
Path('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/data/test').mkdir(exist_ok=True, parents=True)
Path('./word2vec_model/').mkdir(exist_ok=True, parents=True)

#设置参数
word2vec_modell = 'NPs'
Embsize = 150
stride = 1
Embepochs = 50
kmer_len1 = 1
kmer_len2 = 2
kmer_len3 = 3
kmer_len4 = 4
kmer_len5 = 5
kmer_len6 = 6
#定义函数
def Gen_Words(sequences,kmer_len,s):
        out=[]

        for i in sequences:

                kmer_list=[]
                for j in range(0,(len(i)-kmer_len)+1,s):

                            kmer_list.append(i[j:j+kmer_len])

                out.append(kmer_list)

        return out
def read_fasta_file(Filename):
    '''
    used for load fasta data and transformd into numpy.array format
    '''
    fh = open(Filename, 'r')
    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            seq.append(line.replace('\n', '').replace('\r', ''))
    fh.close()
    matrix_data = np.array([list(e) for e in seq])
    #print(matrix_data)
    return seq

def train(sequences,kmer_len):
    print('training word2vec modell')
    document= Gen_Words(sequences,kmer_len,stride)
    #print(document)
    modell = gensim.models.Word2Vec (document, window=int(6), min_count=0, vector_size=Embsize,
                                     workers=multiprocessing.cpu_count(),sg=0,sample=33)
    modell.train(document,total_examples=len(document),epochs=Embepochs)
    modell.save('word2vec_model'+'/'+word2vec_modell+str(kmer_len))
    return document


#训练word2vec并保存模型
all_seq = read_fasta_file('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/data/100total.txt')
document1 = train(all_seq,kmer_len1)
document4 = train(all_seq,kmer_len4)
# document5 = train(all_seq,kmer_len5)
# document6 = train(all_seq,kmer_len6)

model1 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len1))
#model2 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len2))
#model3 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len3))
model4 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len4))

#读取训练集测试集进行训练词向量
x_train1 = pd.read_csv('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/data/train/100X_train.csv')
x_train2 = x_train1['Sequence'].to_numpy()
x_train3 = Gen_Words(x_train2,kmer_len1,stride)
#将训练集通过word2vec--model3进行处理
X_train = []
for i in range(0,len(x_train3)):
    s = []
    for word in x_train3[i]:
       s.append(model1.wv[word])   
    X_train.append(s)
print(np.array(X_train).shape)

#独立数据
x_test1 = pd.read_csv('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/data/test/100X_test.csv')
x_test2 = x_test1['Sequence'].to_numpy()
x_test3 = Gen_Words(x_test2,kmer_len1,stride)
X_test = []
for i in range(0,len(x_test3)):
    s = []
    for word in x_test3[i]:
        s.append(model1.wv[word])   
    X_test.append(s)
print(np.array(X_test).shape)

np.savez('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/data/100_1mer.npz', x_train=np.array(X_train),x_test = np.array(X_test))
