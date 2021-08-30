#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import math
from matplotlib import pyplot

torch.manual_seed(4)
np.random.seed(4)


torch.cuda.empty_cache()


# In[2]:


# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)

input_window = 168 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one
batch_size = 10 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
#         print(self.pe[:x.size(0), :].shape)
#         print(x.shape)
#         y = x + self.pe[:x.size(0), :]
#         print(y.shape)
        return x + self.pe[:x.size(0), :]
          


# In[4]:


class TransAm(nn.Module):
    def __init__(self,feature_size=200,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask




# In[5]:


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)


# In[6]:


def get_data():
    # construct a littel toy dataset
    df = pd.read_csv('/mmdetection3d/Codetrainsformer/API_temp_2015_2021.csv')
#     time = np.array(df['MESS_DATUM'])
#     time = np.arange(0, 400, 0.1)    
    df.set_index(df['MESS_DATUM'], inplace = True)
    df.drop(columns = ['MESS_DATUM'], inplace = True)
    data = df.values
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    amplitude = data[:,0]
#     amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))

    
    #loading weather data from a file
    #from pandas import read_csv
    #series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    #amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    
    
    train_sampels = 20000
    time_until_2019 = 35040
    train_data = amplitude[time_until_2019:time_until_2019 + train_sampels]
    test_data = amplitude[time_until_2019 + train_sampels:]

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack.. 

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence.to(device),test_data.to(device),scaler


# In[7]:


def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]  
#     print(data)
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target



# In[8]:


def train(train_data):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} s | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# In[9]:


def plot_and_loss(eval_model, data_source,epoch,scaler):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            output = eval_model(data)            
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    #test_result = test_result.cpu().numpy() -> no need to detach stuff.. 
    len(test_result)
    test_result_=scaler.inverse_transform(test_result[:10000].unsqueeze(1))
    truth_=scaler.inverse_transform(truth.unsqueeze(1))

    pyplot.figure(figsize=(20, 5))
    pyplot.plot(test_result_,color="red")
    pyplot.plot(truth_[:5000],color="blue")
    pyplot.plot(test_result_-truth_,color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('/mmdetection3d/Codetrainsformer/model_result/transformer-epoch%d.png'%epoch)
    pyplot.close()
    
    return total_loss / i



# In[10]:


# predict the next n steps based on the input data 
def predict_future(eval_model, data_source,steps,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source, 0,1)
    with torch.no_grad():
        for i in range(0, steps):            
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
            
    data = data.cpu().view(-1)
    
    # I used this plot to visualize if the model pics up any long therm struccture within the data. 
    pyplot.figure(figsize=(20, 5))
    pyplot.plot(data,color="red")       
    pyplot.plot(data[:input_window],color="blue")    
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('/mmdetection3d/Codetrainsformer/model_result/transformer-future%d.png'%epoch)
    pyplot.close()
        


# In[11]:


def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


# In[12]:


train_data, val_data, scaler = get_data()
model = TransAm().to(device)

criterion = nn.L1Loss()
lr = 0.005 
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float("inf")
epochs = 20 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    
    if(epoch % 2 == 0):
        val_loss = plot_and_loss(model, val_data,epoch, scaler)
        predict_future(model, val_data,200,epoch)
    else:
        val_loss = evaluate(model, val_data)
   
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
       best_val_loss = val_loss
       best_model = model

    scheduler.step() 


# In[20]:


torch.save(best_model,'/mmdetection3d/Codetrainsformer/model_result/1to1_L1_lr=0.05_win=168_fea=200_layers=1.pkl')


# In[13]:


# criterion = nn.L1Loss()
# val_loss = plot_and_loss(model, val_data, 21, scaler)


# In[14]:


# val_loss


# In[15]:


# test_data = pd.read_csv('newone.csv')
# test_data = test_data[24:224]['Value'].values
# test_data


# In[16]:


test_data = pd.read_csv('newone.csv')
# data = test_data[0:24]['Value'].values
test_data = test_data[24:192]['Value'].values



model.eval() 
total_loss = 0.

data = val_data[-1:] 
# print(data)
data = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size

    
with torch.no_grad():
    for i in range(0, input_window):            
        output = model(data[-input_window:])                        
        data = torch.cat((data, output[-1:]))
            
data = data.cpu().view(-1)
data=scaler.inverse_transform(data.unsqueeze(1))
    
# I used this plot to visualize if the model pics up any long therm struccture within the data. 
pyplot.figure(figsize=(20, 5))
pyplot.plot(data,color="red")
real = data[:input_window]
real = np.append(real,test_data)
pyplot.plot(real,color="blue")
pyplot.grid(True, which='both')
pyplot.axhline(y=0, color='k')
pyplot.savefig('./graph/transformer-future-withlabel_168to168.png')
pyplot.close()
        




