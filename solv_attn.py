#!/usr/bin/env python
# coding: utf-8

# In[29]:


import xlrd 
import pubchempy as pcp
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import seaborn as sns

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec, mol2sentence
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg


# In[30]:


pickle_in = open("fold-0.pickle","rb")
load_dict =  pickle.load(pickle_in)

data_frame = load_dict['shuffled']

data_frame


# In[31]:


# loc = ("MNSol_alldata.xlsx") 
# wb = xlrd.open_workbook(loc) 
# sheet = wb.sheet_by_index(0) 

# solutes = sheet.col_values(2)
# solvents = sheet.col_values(9)
# energies = sheet.col_values(10)

# solute_smis = []
# solvent_smis = []

# for solute in solutes[100:110]:
#     for compound in pcp.get_compounds(solute, 'name'):
#         solute_smis.append(compound.isomeric_smiles)
        
# for solvent in solvents[100:110]:
#     for compound in pcp.get_compounds(solvent, 'name'):
#         solvent_smis.append(compound.isomeric_smiles)

solute_smis = data_frame['SoluteSMILES'].tolist()
solvent_smis = data_frame['SolventSMILES'].tolist()
energies = data_frame['DeltaGsolv'].tolist()

print(len(solute_smis))
print(len(solvent_smis))
print(len(energies))


# In[32]:


solutesmis = [Chem.MolFromSmiles(x) for x in solute_smis]
solventsmis = [Chem.MolFromSmiles(x) for x in solvent_smis]


# In[33]:


solute_sentences = []
solvent_sentences = []

for solute_smi in solutesmis:
    solute_sentences.append(mol2alt_sentence(solute_smi, 1))
    
for solvent_smi in solventsmis:
    solvent_sentences.append(mol2alt_sentence(solvent_smi, 1))
    
print(solute_sentences[6])
print(solvent_sentences[6])


# In[34]:


from gensim.models import word2vec
model = word2vec.Word2Vec.load('model_300dim.pkl')

solute_sequences = []
count = 0
sequences = []
solvent_sequences = []
Gsolv = []
l = len(solute_smis)

for i in range(l):
    flag = 0
    solute_substructures = []
    solvent_substructures = []
    
    for identifier in solute_sentences[i]:
        try:
            solute_substructures.append(model.wv.word_vec(identifier))
        except:
            flag = 1
            break
    
    for identifier in solvent_sentences[i]:
        try:
            solvent_substructures.append(model.wv.word_vec(identifier))
        except:
            flag = 1
            break
            
    if flag == 1:
        count += 1
        continue
    sequences.append((solute_substructures, solvent_substructures))
    solute_sequences.append(solute_substructures)
    solvent_sequences.append(solvent_substructures)
    Gsolv.append(energies[i])
    
    
print(count)
# solute_mol2vecs = [DfVec(x) for x in sentences2vec(solute_sentences, model, unseen='UNK')]
# solvent_mol2vecs = [DfVec(x) for x in sentences2vec(solvent_sentences, model, unseen='UNK')]


# In[35]:


print(len(solute_sequences))
print(len(solvent_sequences))
print(len(Gsolv))
print(len(sequences))

# for i in solute_sequences:
#     print(np.asarray(i).shape)
# print('\n')
# for i in solvent_sequences:
#     print(np.asarray(i).shape)


# In[36]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import math
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, LayerNorm


# In[45]:


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0):
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        self.solv_src_mask = None
        self.solu_src_mask = None
        self.ninp = ninp
        
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        
        self.solv_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.solu_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, solv_src, solu_src):
        if self.solv_src_mask is None or self.solv_src_mask.size(0) != len(solv_src):
            self.solv_src_mask = self._generate_square_subsequent_mask(len(solv_src))
            
        if self.solu_src_mask is None or self.solu_src_mask.size(0) != len(solu_src):
            self.solu_src_mask = self._generate_square_subsequent_mask(len(solu_src))
        
        solv_src = self.pos_encoder(solv_src)
        solu_src = self.pos_encoder(solu_src)
        
        solv_output = self.solv_encoder(solv_src, self.solv_src_mask)
        solu_output = self.solu_encoder(solu_src, self.solu_src_mask)
        
#         print(solv_output.shape)
        
        solv_output = torch.squeeze(solv_output, 1)
        solu_output = torch.squeeze(solu_output, 1)
        
#         print(solv_output.shape)
        
        # shared attention layer
        
        H = solv_output
        G = solu_output
        
        Hsize = H.size(0)
        Gsize = G.size(0)
        a_score = torch.zeros(Hsize, Gsize)
        
        for i in range(Hsize):
            for j in range(Gsize):
#                 print(a_score[i][j], torch.matmul(H[i],torch.t(G[j])))
                a_score[i][j] = torch.matmul(H[i],torch.t(G[j]))
#                 print(a_score[i][j])
                
        
        a = F.softmax(a_score,1)
        
        P = torch.matmul(a, G)
        Q = torch.matmul(torch.t(a), H)
        
        # maxpooling layer
        u = torch.max(H, P)
        v = torch.max(G, Q)
        
        inpu = torch.sum(u, dim=0)
        inpv = torch.sum(v, dim=0)
        
        inp = torch.cat((inpu, inpv), 0)
        
        # mlp
        x = F.relu(self.fc1(inp))
        output = self.fc2(x)
        
#         print(output.shape)
#         print(output)
        
        return output


# In[38]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# In[39]:


ninp = 300 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value


# In[40]:


model = TransformerModel(ninp, nhead, nhid, nlayers, dropout)

# PRINTING MODEL & PARAMETERS 
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[41]:


learning_rate = 0.000002
nest_momentum = 0.009

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=nest_momentum, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

num_epochs = 100
splits = 10

best_model = model
best_loss = float("inf")
losses = []

# Training and 10 fold cross validation
kf = KFold(n_splits = 10)
kf.get_n_splits(sequences)

for train_index, test_index in kf.split(sequences):
    model = TransformerModel(ninp, nhead, nhid, nlayers, dropout)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=nest_momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    seq_tr_va, seq_test = np.asarray(sequences)[train_index], np.asarray(sequences)[test_index]
    E_tr_va, E_test = np.asarray(Gsolv)[train_index], np.asarray(Gsolv)[test_index]
    seq_train, seq_val, E_train, E_val = train_test_split(seq_tr_va, E_tr_va, test_size=0.11)
    
    print("TRAIN:", len(seq_train), "VALIDATION:", len(seq_val), "TEST:", len(seq_test))
    
    #Training
    for epoch in range(num_epochs):   
        model.train()
        for i in range(len(seq_train)):
            solute, solvent = seq_train[i]
            energy = torch.FloatTensor([E_train[i]])
        
            # Forward pass
            solvent = torch.FloatTensor(solvent)
            solute = torch.FloatTensor(solute)
            
            solvent = torch.unsqueeze(solvent, 1)
            solute = torch.unsqueeze(solute, 1)
            
            output = model(solvent, solute)
#             print(output.shape)
            loss = criterion(output, energy)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            
        #Validation
        model.eval()
        outputs = []

        with torch.no_grad():
            for i in range(len(seq_val)):
                solute, solvent = seq_val[i]
                energy = torch.FloatTensor([E_val[i]])
        
                solvent = torch.FloatTensor(solvent)
                solute = torch.FloatTensor(solute)
            
                solvent = torch.unsqueeze(solvent, 1)
                solute = torch.unsqueeze(solute, 1)
            
                output = model(solvent, solute)
            
                outputs.append(output)
                
            val_loss = criterion(torch.squeeze(torch.FloatTensor(E_val), axis=0), torch.squeeze(torch.FloatTensor([outputs]), axis=0))
            print('Validation MSE: {}'.format(val_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                
            scheduler.step()
            
    #Testing
    model.eval()
    outputs = []

    with torch.no_grad():
        for i in range(len(seq_test)):
            solute, solvent = seq_test[i]
            energy = torch.FloatTensor([E_test[i]])
        
            solvent = torch.FloatTensor(solvent)
            solute = torch.FloatTensor(solute)
            
            solvent = torch.unsqueeze(solvent, 1)
            solute = torch.unsqueeze(solute, 1)
            
            output = best_model(solvent, solute)
            
            outputs.append(output)
                
        test_loss = criterion(torch.squeeze(torch.FloatTensor(E_test), axis=0), torch.squeeze(torch.FloatTensor([outputs]), axis=0))
        print('Test MSE: {}'.format(test_loss))
        losses.append(test_loss)
    


# In[ ]:


print(losses)

with open('losses.txt', 'a') as f:
    for item in losses:
        f.write("%s\n" % item)


# In[ ]:


# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')

