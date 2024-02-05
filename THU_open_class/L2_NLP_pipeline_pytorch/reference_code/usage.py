import torch
from torch import nn
import data

corpus = data.Corpus("../data/glue-sst2")

def batchify(data, bsz):
    nbatch = len(data) // bsz
    data = data[:nbatch * bsz]
    batches = []
    target = []
    temp = []
    cnt = 0
    for item in data:
        x,y = item
        target.append(y)
        temp = temp + x.tolist()
        cnt+=1
        if(cnt==bsz):
            batches.append((torch.tensor(temp).view(bsz,10).t(),target))
            temp = []
            target = []
            cnt = 0
    return batches

eval_batch_size = 10

model = torch.load('model.pt')

total_test_data = len(corpus.test)

test_data = batchify(corpus.test, eval_batch_size)

criterion = nn.CrossEntropyLoss()

def evaluate(data_source): 
    acc = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data_source), 1):
            data, target = data_source[i]
            output = model(data)
            output = output.argmax(dim=1)
            for j in range(eval_batch_size):
                if(output[j]==target[j]):
                    acc += 1
            # 如果你需要预测输出 可以使用以下print代码
            # for item in output:
            #     # if item == 1 :
            #     #     # print("positive",end=" ")
            #     # else:
            #     #     # print("negative",end=" ")
    return (acc / total_test_data)

test = evaluate(test_data)

print(test*100,"% accuracy")