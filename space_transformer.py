import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import space_finder
from data import tit_space
import time

# Hyperparameter----

batch_size = 256
lr = 1e-2
epoch = 100
batch_eval_inter = 100
eval_train_test = 10
n_embd = 512
n_heads = 8
n_layers = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_test = 10

# ------------------

# Splitting the data
tit_data = tit_space()
train_dataset, test_dataset = random_split(tit_data, [0.9, 0.1])

# Loading the data
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# Initialize the model
model = space_finder(n_embd, n_heads, n_layers)
model = model.to(device)

# loss ,Optimizer, Scheduler
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)


#Training the model
start = time.time()

import time
from sklearn.metrics import accuracy_score

def train_loop(i, see_batch_loss = False):
    model.train()
    total_loss = 0
    y_true = []
    y_preds = []
    for batch, (data, label) in enumerate(train_loader):
        data , label = data.to(device), label.to(device)
        # print(data.shape)
        logits = model(data)
        # preds = preds.argmax(dim = -1)
        # print(logits)
        optimizer.zero_grad()
        loss = loss_fn(logits.view(-1), label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.sigmoid(logits).view(-1) > 0.5
        y_true.extend(label.cpu().tolist())
        y_preds.extend(preds.detach().cpu().tolist())
        
        if see_batch_loss:
            if batch%batch_eval_inter == 0:
                print(f'Batch_Loss_{batch} : {loss.item()}')

    if i%eval_train_test==0:
        val_loss, val_acc = test_loop(test_loader)
        print(f'Epoch {i+1}: train_loss: {(total_loss/len(train_loader)):.4f} | train_acc: {(accuracy_score(y_true, y_preds)):.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}')
        


def test_loop(dataset):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        y_true = []
        y_preds = []
        for data, label in dataset:
            data , label = data.to(device), label.to(device)
            logits = model(data)
    
            loss = loss_fn(logits.view(-1), label)
            
            total_loss+=loss.item()
            preds = torch.sigmoid(logits).view(-1) > 0.5
            y_true.extend(label.cpu().tolist())
            y_preds.extend(preds.detach().cpu().tolist())
                              
    return total_loss/len(test_loader), accuracy_score(y_true, y_preds)
    # print(f'val_loss: {total_loss/len(test_loader)}, val_acc: {accuracy_score(y_true, y_preds)}')  

if __name__ == '__main__':

    for epoch in range(epoch): 
        train_loop(epoch)
        # break
        
    end = time.time()

    print(f'Total_time: {end-start}')

