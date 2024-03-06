"""
Sources for the code:
1. https://github.com/sleebapaul/gospel_of_rnn/blob/master/pytorch_LSTM.ipynb
2. https://github.com/pytorch/examples/tree/master/word_language_model

The original Dataset provided has been reduced and 120,000 lines have been used for the Assignment. The dataset was then split further into Training, Validation and Test
"""

from os import path
import torch
torch.manual_seed(1111)
import math
import os
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_

import argparse

 
#prints the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print("\n")

parser = argparse.ArgumentParser(description='Word Language Model using LSTM')

parser.add_argument('--data', type=str, default='/home/csgrads/rao00134/smalltext',
                    help='location of the data corpus')

parser.add_argument('-f')

args = parser.parse_args()


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx = 0
    
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
        self.train = self.tokenize(os.path.join(path, 'smalltext.train2.txt'))
        self.valid = self.tokenize(os.path.join(path, 'smalltext.val2.txt'))
        self.test = self.tokenize(os.path.join(path, 'smalltext.test2.txt'))


    def tokenize(self, path):
        """Tokenizes a text file."""
        
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids



def batchify(data, bsz):
    """
    Batch the data with size bsz
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    """
    Select seq_length long batches at once for training
    Data and Targets will be differed in index by one
    """
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    For truncated backpropagation 
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.LSTM = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.LSTM(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


def evaluate(data_source, eval_batch_size):
    """
    Evaluates the performance of the trained model in input data source
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = vocab_size
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_length):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    """
    Training 
    """
    # Turn on training mode which enables dropout.
    
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = vocab_size
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_length)):
        data, targets = get_batch(train_data, i)
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} |'.format(
                epoch, batch, len(train_data) // seq_length, learning_rate,
                elapsed * 1000 /log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def get_warmup_state(data_source):
    """
    Starting hidden states as zeros might not deliver the context 
    So a warm up is on a desired primer text 
    Returns the hidden state for actual generation 
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = vocab_size
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_length):
            data, targets = get_batch(data_source, i)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)         
    return hidden


# Hyper-parameters

embed_size = 300
hidden_size = 1024
num_layers = 2
num_epochs = 5
batch_size = 30
seq_length = 35
learning_rate = 20.0
dropout_value = 0.4
log_interval = 100
eval_batch_size = 10


corpus = Corpus(args.data)


#batchify the datasets
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

vocab_size = len(corpus.dictionary)

#prints the shape of the training, validation and test data
print("\n")
print("Shape of batchified training data: ", train_data.shape)
print("Shape of batchified validation data: ", val_data.shape)
print("Shape of batchified testing data: ", test_data.shape)
print("\n")


# Define model for training 

model = RNNModel(ntoken=vocab_size, ninp=embed_size, nhid=hidden_size, nlayers=num_layers, dropout=dropout_value).to(device)
criterion = nn.CrossEntropyLoss()


# Start training the model

best_val_loss = None
training_loss = []
validation_loss = []


try:
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        
        # Start training for one epoch
        train()
        
        # Get and store validation and training losses 
        val_loss = evaluate(val_data, eval_batch_size)
        tr_loss = evaluate(train_data, batch_size)
        
        
        training_loss.append(tr_loss)
        validation_loss.append(val_loss)
        
        print('-' * 122)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | training loss {:5.2f} | training ppl {:8.2f} |'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss), tr_loss, math.exp(tr_loss)))
        print('-' * 122)
        
        
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), "/home/csgrads/rao00134/model.pt") #path to save the model
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            learning_rate = learning_rate /1.5
except KeyboardInterrupt:
    #Exits training if their is a keyboard interrupt
    print('-' * 122)
    print('Exiting from training early')
