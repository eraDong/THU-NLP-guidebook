import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.seq_len = 10
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.dev = self.tokenize(os.path.join(path, 'dev.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        self.dictionary.add_word('<pad>')  
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()  
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r', encoding="utf8") as f:
            idss = []
            labels = []
            for line in f:
                words = line.split() 
                ids = []

                #判断标签
                if(words[-1]=="negative"):
                    labels.append(0)
                else:
                    labels.append(1)
                
                #剥夺标签
                words = words[:-1]
                
                #增长序列
                if(len(words)>self.seq_len):
                    words = words[:self.seq_len]
                else:
                    while(len(words)<self.seq_len):
                        words.append('<pad>')

                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            
            ids = torch.stack(idss)
        samples = [(ids, label) for ids, label in zip(idss, labels)]
        return samples
