from torch.utils.data import DataLoader
from dataset import MelanomaDataset
from params import *

def data_loader(df_train, df_valid, transform_tuple= (None,None), batch_size=None):
  dataset = {'train': MelanomaDataset(df_train['path'], df_train['target'],transform=transform_tuple[0]),
             'val': MelanomaDataset(df_valid['path'], df_valid['target'],transform=transform_tuple[1])}
  train_loader_ = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=params['num_workers'])
  val_loader_ = DataLoader(dataset['val'], batch_size=batch_size*2, pin_memory=True, num_workers = params['num_workers'])
  return train_loader_, val_loader_