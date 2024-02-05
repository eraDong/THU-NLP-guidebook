'''
Author: eraDong 115410761+eraDong@users.noreply.github.com
Date: 2024-02-04 21:18:24
LastEditors: eraDong 115410761+eraDong@users.noreply.github.com
LastEditTime: 2024-02-05 16:31:18
FilePath: \课程练习+代码\L2_NLP_pipeline_pytorch\my_code\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: eraDong 115410761+eraDong@users.noreply.github.com
Date: 2024-02-04 12:51:13
LastEditors: eraDong 115410761+eraDong@users.noreply.github.com
LastEditTime: 2024-02-04 22:06:18
FilePath: \课程练习+代码\L2_NLP_pipeline_pytorch\annotated_code\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn


import data
import model
import sys

from torch.utils.data import DataLoader

# 为控制台参数设置一系列操作
parser = argparse.ArgumentParser(description='PyTorch glue-sst2 RNN/LSTM/GRU/Transformer Language Model')

parser.add_argument('--data', type=str, default='../data/glue-sst2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,  # 你可能需要调整它
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()

# 设置随机种子保证结果的可重复性
torch.manual_seed(args.seed)

# cuda设置
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

corpus = data.Corpus(args.data)


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
train_data = batchify(corpus.train, args.batch_size)

val_data = batchify(corpus.dev, eval_batch_size)

test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)
if args.model == 'LSTM':
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):  
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, len(data_source), args.bptt):
            data, targets = data_source[i]
            output = model(data)
            total_loss += len(data) * criterion(output, torch.tensor(targets)).item()
    return total_loss / (len(data_source) - 1)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)    
    for batch,i in enumerate(range(0, len(train_data) , args.bptt)):
        data, targets = train_data[i]
        optimizer.zero_grad()
        output = model(data) 
        loss = criterion(output, torch.tensor(targets)) 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step() 
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))) 
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)



