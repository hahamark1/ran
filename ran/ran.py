import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN


class RAN(nn.Module):

    def __init__(self, input_size, hidden_size, nlayers=1, dropout=0.5, cuda = True):
        super().__init__()
        # if nlayers > 1:
        #     raise NotImplementedError("TODO: nlayers > 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout



        self.w_cx = [nn.Parameter(torch.Tensor(hidden_size, input_size)) for i in range(nlayers)]
        self.w_ic = [nn.Parameter(torch.Tensor(hidden_size, hidden_size)) for i in range(nlayers)]
        self.w_ix = [nn.Parameter(torch.Tensor(hidden_size, input_size)) for i in range(nlayers)]
        self.w_fc = [nn.Parameter(torch.Tensor(hidden_size, hidden_size)) for i in range(nlayers)]
        self.w_fx = [nn.Parameter(torch.Tensor(hidden_size, input_size)) for i in range(nlayers)]

        self.b_cx = [nn.Parameter(torch.Tensor(hidden_size)) for i in range(nlayers)]
        self.b_ic = [nn.Parameter(torch.Tensor(hidden_size)) for i in range(nlayers)]
        self.b_ix = [nn.Parameter(torch.Tensor(hidden_size)) for i in range(nlayers)]
        self.b_fc = [nn.Parameter(torch.Tensor(hidden_size)) for i in range(nlayers)]
        self.b_fx = [nn.Parameter(torch.Tensor(hidden_size)) for i in range(nlayers)]

        self.weights = [(self.w_cx[i], self.w_ic[i], self.w_ix[i], self.w_fc[i], self.w_fx[i]) for i in range(nlayers)]
        for weight in self.weights:
            for w in weight:
                init.xavier_uniform(w)
                w.cuda()

        self.biases = [(self.b_cx[i], self.b_ic[i], self.b_ix[i], self.b_fc[i], self.b_fx[i]) for i in range(nlayers)]
        for bias in self.biases:
            for b in bias:
                b.data.fill_(0)
                b.cuda()
        if cuda:
            self.cuda()
    def forward(self, input, hidden):
        layer = (Recurrent(RANCell), )
        func = StackedRNN(layer, self.nlayers, dropout=self.dropout)
        hidden, output = func(input, hidden, tuple(((self.weights[i], self.biases[i]) for i in range(self.nlayers))))
        return output, hidden


def RANCell(input, hidden, weights, biases):
    print(len(weights))
    w_cx, w_ic, w_ix, w_fc, w_fx = weights
    b_cx, b_ic, b_ix, b_fc, b_fx = biases
    print(input.is_cuda)
    input = input.cpu()
    print(w_cx.is_cuda)
    print(w_cx.size())
    ctilde_t = F.linear(input, w_cx, b_cx)
    i_t = F.sigmoid(F.linear(hidden, w_ic, b_ic) + F.linear(h_t, w_ix, b_ix))
    f_t = F.sigmoid(F.linear(hidden, w_fc, b_fc) + F.linear(h_t, w_fx, b_fx))
    c_t = i_t * ctilde_t + f_t * hidden
    h_t = F.tanh(c_t)
    return h_t,c_t
