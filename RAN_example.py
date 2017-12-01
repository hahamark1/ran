# coding: utf-8

"""
CBOW

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""

from collections import defaultdict
import time
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as function

torch.manual_seed(42)


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
i2w = {}
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            words = line.lower().strip().split(" ") + ['</s>']
            for word in words:
                if word not in w2i:
                    w2i[word]
                    i2w[len(i2w)] = word
            yield ([[w2i[words[x]], w2i[words[x+1]]] for x in range(len(words)-1)])



# Read in the data
train = list(read_dataset("try.txt"))
print(train)
w2i = defaultdict(lambda: UNK, w2i)
print(w2i)
print(i2w)
dev = list(read_dataset("text.txt"))
nwords = len(w2i)
ntags = len(t2i)

hidden_size = nwords

class RAN(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(RAN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, input_dim)
        self.w_newState = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.w_input_old = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.w_input_new = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.w_forget_old = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.w_forget_new = nn.Parameter(torch.Tensor(hidden_dim, input_dim))

        self.b_input = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_forget = nn.Parameter(torch.Tensor(hidden_dim))

        self.weights = self.w_newState, self.w_input_old, self.w_input_new, self.w_forget_old, self.w_forget_new
        self.biases = self.b_input, self.b_forget
        for w in self.weights:
            init.xavier_uniform(w)
        for b in self.biases:
            init.uniform(b)
        self.nlayers = 1
        self.nhid = hidden_size
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)


    def forward(self, input, hidden):
        embeds = self.embeddings(input)
        c_tilde = function.linear(embeds, self.w_newState)
        i_t = function.sigmoid(function.linear(hidden, self.w_input_old) + function.linear(embeds, self.w_input_new) + self.b_input)
        f_t = function.sigmoid(function.linear(hidden, self.w_forget_old) + function.linear(embeds, self.w_forget_new) + self.b_forget)
        c_t = i_t * c_tilde + f_t * hidden
        # size of all inbetween states should be euqal to c_tilde
        # hidden size is not correct, how is it defined
        h_t = function.tanh(c_t)
        c_t_d = self.decoder(c_t.view(c_t.size(0)*c_t.size(1)))
        return c_t_d.view(c_t.size(0), c_t.size(1)), h_t

    def target_embeddings(self, input):
        embeds = self.embeddings(input)
        return embeds

    def init_hidden(self):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, self.nhid).zero_()).cuda()



model = RAN(nwords, hidden_size, hidden_size, ntags)
model.cuda()
print(model)

def repackage_hidden(h):
    """Wraps hidden states in new Variables,
        to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0
    for line in data:
        hidden = model.init_hidden().cuda()
        for word, next_word in line:
            lookup_tensor = Variable(torch.LongTensor([word])).cuda()
            scores, hidden = model(lookup_tensor, hidden)
            indices = np.argmax(scores.cpu().data.numpy())
            print(indices)
            # predict = scores.data.numpy().argmax(axis=1)[0]
            # target = Variable(torch.LongTensor([next_word]), requires_grad=False).cuda()
            # target = model.target_embeddings(target)

            predict = i2w[indices-1]
            print(indices, next_word)
            if indices == next_word:
                correct += 1

    return correct, len(data), correct/len(data)

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

for ITER in range(100):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    hidden = model.init_hidden().cuda()
    print(len(train))
    i = 0
    for line in train:
        i +=1
        if i % 100 == 0 :
            print(i)
        hidden = model.init_hidden()
        for word, next_word in line:
            # forward pass
            lookup_tensor = Variable(torch.LongTensor([word])).cuda()
            scores, hidden = model(lookup_tensor, hidden)
            target = Variable(torch.LongTensor([next_word]), requires_grad=False).cuda()
            target = model.target_embeddings(target)
            output = mse_loss(scores, target)
            train_loss += output.data[0]

            # backward pass
            model.zero_grad()
            output.backward(retain_graph=True)

            # update weights
            optimizer.step()


    print("iter %r: train loss/sent=%.4f, time=%.2fs" %
          (ITER, train_loss/len(train), time.time()-start))

# evaluate
_, _, acc = evaluate(model, dev)
print("iter %r: test acc=%.4f" % (ITER, acc))
