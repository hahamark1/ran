import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN


class RAN(nn.Module):

    def __init__(self, input_size, hidden_size, nlayers=1, dropout=0.5):
        super().__init__()
        # if nlayers > 1:
        #     raise NotImplementedError("TODO: nlayers > 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.weights_dict = {}
        # for i in range(nlayers):
        #     self.weights_dict[i] =
        self.w_cx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_ic = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_ix = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_fc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_fx = nn.Parameter(torch.Tensor(hidden_size, input_size))


        self.w_cx_2 = [nn.Parameter(torch.Tensor(hidden_size, hidden_size)).cuda() for i in range(nlayers-1)]
        self.w_ic_2 = [nn.Parameter(torch.Tensor(hidden_size, hidden_size)).cuda() for i in range(nlayers-1)]
        self.w_ix_2 = [nn.Parameter(torch.Tensor(hidden_size, hidden_size)).cuda() for i in range(nlayers-1)]
        self.w_fc_2 = [nn.Parameter(torch.Tensor(hidden_size, hidden_size)).cuda() for i in range(nlayers-1)]
        self.w_fx_2 = [nn.Parameter(torch.Tensor(hidden_size, hidden_size)).cuda() for i in range(nlayers-1)]

        self.b_cx = [nn.Parameter(torch.Tensor(hidden_size)).cuda() for i in range(nlayers)]
        self.b_ic = [nn.Parameter(torch.Tensor(hidden_size)).cuda() for i in range(nlayers)]
        self.b_ix = [nn.Parameter(torch.Tensor(hidden_size)).cuda() for i in range(nlayers)]
        self.b_fc = [nn.Parameter(torch.Tensor(hidden_size)).cuda() for i in range(nlayers)]
        self.b_fx = [nn.Parameter(torch.Tensor(hidden_size)).cuda() for i in range(nlayers)]

        self.weights_1 = self.w_cx, self.w_ic, self.w_ix, self.w_fc, self.w_fx
        self.weights_2 = [tuple([self.w_cx_2[i], self.w_ic_2[i], self.w_ix_2[i], self.w_fc_2[i], self.w_fx_2[i]]) for i in range(nlayers-1)]

        for w in self.weights_1:
            init.xavier_uniform(w)

        for weight in self.weights_2:
            for w in weight:
                init.xavier_uniform(w)

        self.biases = [tuple([self.b_cx[i], self.b_ic[i], self.b_ix[i], self.b_fc[i], self.b_fx[i]]) for i in range(nlayers)]
        for bias in self.biases:
            for b in bias:
                b.data.fill_(0)

    def forward(self, input, hidden):
        layer = (Recurrent(RANCell), )
        func = StackedRNN(layer, self.nlayers, dropout=self.dropout)
        weight_layer = tuple([tuple([self.weights_1, self.biases[0]])]) + tuple(tuple([self.weights_2[i], self.biases[i+1]]) for i in range(self.nlayers-1))
        hidden, output = func(input, hidden, weight_layer)
        return output, hidden

def RANCell(input, hidden, weights, biases):
    # print(weights.size
    w_cx, w_ic, w_ix, w_fc, w_fx = weights
    b_cx, b_ic, b_ix, b_fc, b_fx = biases
    # print('sizes', w_cx.size(), input.size(), hidden.size())
    ctilde_t = F.linear(input, w_cx, b_cx)
    i_t = F.sigmoid(F.linear(hidden, w_ic, b_ic) + F.linear(input, w_ix, b_ix))
    f_t = F.sigmoid(F.linear(hidden, w_fc, b_fc) + F.linear(input, w_fx, b_fx))
    c_t = i_t * ctilde_t + f_t * hidden
    h_t = F.tanh(c_t)
    return h_t
