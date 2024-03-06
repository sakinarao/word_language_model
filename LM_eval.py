from os import path
import torch
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Word Language Model using LSTM')

parser.add_argument('--data', type=str, default='/home/csgrads/rao00134/smalltext',
                    help='location of the data corpus')

parser.add_argument('-f')

args = parser.parse_args()

"""
Helper Functions
"""
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
        #assert os.path.exists(path)
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

class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    PyTorch provides the facility to write custom models
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

#Hyperparameters
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

#Batchify Datasets
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

vocab_size = len(corpus.dictionary)


#Defining the model and loading the states
model = RNNModel(ntoken=vocab_size, ninp=embed_size, nhid=hidden_size, nlayers=num_layers, dropout=dropout_value).to(device)
model.load_state_dict(torch.load("/home/csgrads/rao00134/model.pt"))

criterion = nn.CrossEntropyLoss()

#Run on Validation Data
val_loss = evaluate(val_data, eval_batch_size)

print('=' * 89)
print('| Validation loss {:5.2f} | validation ppl {:8.2f}'.format(
    val_loss, math.exp(val_loss)))
print('=' * 89)

#Run on Test data
test_loss = evaluate(test_data, eval_batch_size)

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

#Run on Train data
train_loss = evaluate(train_data, batch_size)

print('| Train loss {:5.2f} | Training ppl {:8.2f}'.format(
    train_loss, math.exp(train_loss)))
print('=' * 89)

#ANALYSIS QUESTIONS

"""

1. Describe the language model that gave you the best results on the validation data,
and that you used on the test data.  What was the structure of that language model,
and what hyperparameters did you set (and what values did you use to set them)?
What sources gave you "inspiration" for this model? (200 word minimum, show word count)

The Language Model that gave the best results on the validation data used the LSTM network
and I used the same on the test data. The model consists of a dropout layer, encoder,
LSTM, decoder. The final hyperparameters that produced the best results were as follows:
(I trained the model only for 5 epochs as it was taking a lot of time to train)
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

I ran the model multiple times changing the hyperparameters. For the first time I ran
the model with embedding size 100, hidde size 100, num_layers 1 and dropout 0. For the
next run, I increased hidden size to 200 which gave better results. So I increased the
hidden size even more to 1024, num_layers to 2 and dropout to 0.4 and I got much better
results with these hyperparameters. 

I found multiple sources which helped me solve the problem and implement this Language Model.
Some of which are listed below:
https://github.com/pytorch/examples/tree/master/word_language_model 
This one really gives an overall understanding of the Word Language Model. All the files
are documented well making it easy to understand the task and help implement your own model.

I found another good model which helped me in implementing mine:
https://github.com/sleebapaul/gospel_of_rnn 


Word Count: 227

"""


"""

2. Explain what Perplexity is, and what it tells us about a language model (in general). 
Compare the Perplexity on your validation data, test data, and training data. How did they
differ? What conclusions can we draw from this result? (200 word minimum, show word count)

Perplexity is an evaluation metric for Language Models. It is an Intrinsic Evaluation
Method which nvolves finding some metric to evaluate the language model itself, not
taking into account the specific tasks it’s going to be used for.
Perplexity can also be defined as the exponential of the cross-entropy, where the
cross-entropy indicates the average number of bits needed to encode one word, and
perplexity is the number of words that can be encoded with those bit.

Perplexity measures how good a Language Model is. If we have a perplexity of 100, it means
that whenever the model is trying to guess the next word it is as confused as if it had
to pick between 100 words. Lower Perplexity is better.

The training perplexity was the lowest and the test perplexity was the highest. While 
training the model we can see the perplexity is really high for the initial steps and it 
decreases with each epoch as the model keeps on learning on the same data. The test perplexity 
is high compared to the training data as we are testing the trained model on new data. It is
still less than the initial values while training which means the model has learned.  

The following link explains perplexity very well:
https://towardsdatascience.com/perplexity-in-language-models-87a196019a94

Word Count: 207

"""


"""

3. How does the Perplexity of your final model on your test data compare to published 
Neural Language Model perplexity results? Find one paper that reports the results of a 
NLM using perplexity and compare your results to those.  Provide a citation to that paper 
in your response. How does your Perplexity compare to the published result? What kind of 
model and data did the published result use? What do you think accounts for the difference 
in the perplexity you see? (200 word minimum, show word count) 

The perplexity of the Final Model is not that great compared to the published NeuraL Language 
Models but still comparable. I would say that the model can give better results if I figure out 
more factors affecting it. I have tuned the model as much as I could but I guess better 
results can be produced usimg this model. 

I found a paper on NLM that uses LSTM and reports the results using perplexity. The perplexity 
they reported is good compared to mine. They used LSTM and fixed the hidden size to 100. They 
also compared the results to the vanilla RNN but got the best results with LSTM. They reported 
that The simpler “vanilla” RNN network showed to be unstable whilte they were changing the input 
of the model . They reported the perplexity on two datasets: Penn Treebank corpus (144) and MALACH 
corpus (99) and the perplexity of my model on the test data is 309. 

I guess I can get better results if I try and change more hyperparameters and if I could train 
the model on the larger corpus. I had to reduce the size of the corpus as it was taking really 
long to run even 1 epoch. So, if I could figure that out, I could have obtained better results. 

Link to the paper: https://link.springer.com/chapter/10.1007/978-3-319-25789-1_25#Sec5

Word Count: 221

"""
