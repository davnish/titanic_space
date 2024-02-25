import torch
from torch.utils.data import Dataset
import pandas as pd

class tit_space(Dataset):
    def __init__(self, to_ = 'train'):
        df = pd.read_csv(f'spaceship-titanic/{to_}.csv')

        df.dropna(inplace = True)
        df.drop(['Name', 'PassengerId', 'Cabin'], axis = 1, inplace = True)
        df = pd.get_dummies(df, columns = ['HomePlanet', 'Destination'], drop_first = True)
        df.replace([False, True], [0,1], inplace = True)
        df = (df-df.min())/(df.max() - df.min())
        # print(df)
        self.label = torch.tensor(df.pop('Transported').to_numpy(), dtype = torch.float)
        self.df = torch.tensor(df.to_numpy(), dtype = torch.float)
    
    def __getitem__(self, idx):
        return self.df[idx], self.label[idx]
    
    def __len__(self):
        return self.df.shape[0]
