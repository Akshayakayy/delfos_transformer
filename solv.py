#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


pickle_in = open("fold-0.pickle","rb")
load_dict =  pickle.load(pickle_in)

data_frame = load_dict['shuffled']

data_frame


# In[3]:


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


# In[4]:


solutesmis = [Chem.MolFromSmiles(x) for x in solute_smis]
solventsmis = [Chem.MolFromSmiles(x) for x in solvent_smis]


# In[5]:


solute_sentences = []
solvent_sentences = []

for solute_smi in solutesmis:
    solute_sentences.append(mol2alt_sentence(solute_smi, 1))
    
for solvent_smi in solventsmis:
    solvent_sentences.append(mol2alt_sentence(solvent_smi, 1))
    
print(solute_sentences[6])
print(solvent_sentences[6])


# In[6]:


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


# In[7]:


print(len(solute_sequences))
print(len(solvent_sequences))
print(len(Gsolv))
print(len(sequences))

# for i in solute_sequences:
#     print(np.asarray(i).shape)
# print('\n')
# for i in solvent_sequences:
#     print(np.asarray(i).shape)


# In[8]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


# In[9]:


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, batch_size, bidirectional = True):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.bidirectional = bidirectional
        self.batch_size = batch_size

        self.solv_lstm = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=True, bidirectional = bidirectional)
        self.solu_lstm = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=True, bidirectional = bidirectional)

        self.fc1 = nn.Linear(hidden_dim*2*2, 2000)
        self.fc2 = nn.Linear(2000, output_dim)
        
    def forward(self, inputs_solv, inputs_solu):
        # Initialize hidden state
        h0_solv = torch.zeros(self.num_layer*(1 + int(self.bidirectional)), self.batch_size, self.hidden_dim)
        h0_solu = torch.zeros(self.num_layer*(1 + int(self.bidirectional)), self.batch_size, self.hidden_dim)

        # Initialize cell state
        c0_solv = torch.zeros(self.num_layer*(1 + int(self.bidirectional)), self.batch_size, self.hidden_dim)
        c0_solu = torch.zeros(self.num_layer*(1 + int(self.bidirectional)), self.batch_size, self.hidden_dim)
        
        H, _ = self.solv_lstm(inputs_solv, (h0_solv, c0_solv)) # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        G, _ = self.solu_lstm(inputs_solu, (h0_solu, c0_solu))
        
        # shared attention layer
        Hsize = H.size(1)
        Gsize = G.size(1)
        a_score = torch.zeros(Hsize, Gsize)
        
        for i in range(Hsize):
            for j in range(Gsize):
                a_score[i][j] = torch.matmul(H[0][i],torch.t(G[0][j]))
                
        
        a = F.softmax(a_score,1)
        
        G = torch.squeeze(G, axis=0)
        H = torch.squeeze(H, axis=0)
        
        P = torch.matmul(a, G)
        Q = torch.matmul(torch.t(a), H)
        
        # maxpooling layer
        u = torch.max(H, P)
        v = torch.max(G, Q)
        
        inpu = torch.sum(u, dim=0)
        inpv = torch.sum(v, dim=0)
        
        inp = torch.cat((inpu, inpv), 0)
        
        # mlp
        x = F.relu(self.fc1(inp)) #number of layers? concat? sum?
        solvE = self.fc2(x)
      
        return solvE


# In[10]:


bidirectional = True

input_dim = 300
hidden_dim = 150
layer_dim = 1
output_dim = 1

batch_size = 1


# In[11]:


model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, batch_size)

# PRINTING MODEL & PARAMETERS 
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[21]:


learning_rate = 0.0002
nest_momentum = 0.9

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=nest_momentum, nesterov=True)


# In[31]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

num_epochs = 100
splits = 10

best_model = model
best_loss = 10000
losses = []

# Training and 10 fold cross validation
kf = KFold(n_splits = 10)
kf.get_n_splits(sequences)

for train_index, test_index in kf.split(sequences):
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, batch_size)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=nest_momentum, nesterov=True)
    
    seq_tr_va, seq_test = np.asarray(sequences)[train_index], np.asarray(sequences)[test_index]
    E_tr_va, E_test = np.asarray(Gsolv)[train_index], np.asarray(Gsolv)[test_index]
    seq_train, seq_val, E_train, E_val = train_test_split(seq_tr_va, E_tr_va, test_size=0.11)
    
    print("TRAIN:", len(seq_train), "VALIDATION:", len(seq_val), "TEST:", len(seq_test))
    
    #Training
    for epoch in range(num_epochs):       
        for i in range(len(seq_train)):
            solute, solvent = seq_train[i]
            energy = torch.FloatTensor([E_train[i]])
        
            # Forward pass
            output = model((torch.FloatTensor([solvent])), (torch.FloatTensor([solute])))
            loss = criterion(output, energy)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            
        #Validation
        outputs = []

        with torch.no_grad():
            for i in range(len(seq_val)):
                solute, solvent = seq_val[i]
                energy = torch.FloatTensor([E_val[i]])
        
                output = model((torch.FloatTensor([solvent])), (torch.FloatTensor([solute])))
                outputs.append(output)
                
            val_loss = criterion(torch.squeeze(torch.FloatTensor(E_val), axis=0), torch.squeeze(torch.FloatTensor([outputs]), axis=0))
            print('Validation MSE: {}'.format(val_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
            
    #Testing
    outputs = []

    with torch.no_grad():
        for i in range(len(seq_test)):
            solute, solvent = seq_test[i]
            energy = torch.FloatTensor([E_test[i]])
        
            output = best_model((torch.FloatTensor([solvent])), (torch.FloatTensor([solute])))
            outputs.append(output)
                
        test_loss = criterion(torch.squeeze(torch.FloatTensor(E_test), axis=0), torch.squeeze(torch.FloatTensor([outputs]), axis=0))
        print('Test MSE: {}'.format(test_loss))
        losses.append(test_loss)
    


# In[61]:


print(losses)

with open('losses.txt', 'a') as f:
    for item in losses:
        f.write("%s\n" % item)


# In[ ]:


# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')

