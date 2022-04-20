from re import A
from torch import nn
from tqdm import tqdm
import time

from visuals import save_metrics, save_epoch_lr
from params import *
from dataframe import create_df
from dataloader import data_loader

from model import MelanomaClassifier
from lr_scheduler import LRSchedulerPlateau
from lr_scheduler_cosine import LRCosineScheduler
from focal_loss import FocalLoss

from early_stopping import EarlyStopping
from metrics import get_lr
from augmentations import transform_aug, transform_normalize
# from augmentations2  import 

from visuals import save_metrics

import pandas as pd
from sklearn.utils import class_weight
import numpy as np

from fit import fit, validate

from timm.optim import AdamP

from torch.utils.data import DataLoader
from get_weights import get_weights
import numpy as np

aug_transform = transform_aug(params['im_size']) if params['augmentation'] else (None,None)
# transform_normalize(params['im_size']

df, df_val = create_df(dir=params['path'], file_name='train')
print("df:",df.shape)
print("df_val:",df_val.shape)

train_loader, val_loader = data_loader(df, df_val, transform_tuple = aug_transform, batch_size = params['batch_size'])

train_loss, train_acc, train_f1, train_auc, val_loss, val_acc, val_f1, val_auc  = [], [], [], [], [], [], [], []
fit_lr = []
start = time.time()
best_=0

model = MelanomaClassifier(params['model'],n_class=8,pretrained=True)

whole = pd.concat([df, df_val], sort=False)
class_weights_ = get_weights(whole['target'], num_class=8)

criterion = nn.CrossEntropyLoss(weight=class_weights_)
#criterion = nn.CrossEntropyLoss()
if params['focal_loss']:
  #whole = pd.concat([df, df_val], sort=False)
  #weights = list(whole['target'].value_counts().sort_index()/len(whole))
  #criterion = FocalLoss(weight=torch.tensor(weights).to(params['device']),gamma=0.5)
  #samples_per_cls = None

  whole = pd.concat([df, df_val], sort=False)
  samples_per_cls = list(whole['target'].value_counts().sort_index())
else:
  samples_per_cls = None


optimizer = AdamP(model.parameters(), lr=params['lr'])

if params['cosine_scheduler']:
  #lr_scheduler = LRCosineScheduler(optimizer, t_initial=params['epochs'])
  lr_scheduler = LRCosineScheduler(optimizer, t_initial=params['epochs']/20, warmup_t=0, warmup_lr_init=0, cycle_limit=20, cycle_decay=0.8)
else:
  lr_scheduler = LRSchedulerPlateau(optimizer)

early_stopping = EarlyStopping()

model.to(params['device'])


for epoch in range(1, params["epochs"] + 1):
  lr_ = get_lr(optimizer)
  print('learning rate:',lr_)
  train_epoch_loss, train_epoch_acc, train_epoch_f1, train_epoch_auc = fit(train_loader, model, criterion, optimizer, epoch, params, samples_per_cls)
  val_epoch_loss, val_epoch_acc, val_epoch_f1, val_epoch_auc = validate(val_loader, model, criterion, epoch, params, samples_per_cls)
  train_loss.append(train_epoch_loss);train_acc.append(train_epoch_acc);train_f1.append(train_epoch_f1);train_auc.append(train_epoch_auc)
  val_loss.append(val_epoch_loss);val_acc.append(val_epoch_acc);val_f1.append(val_epoch_f1);val_auc.append(val_epoch_auc)
  try:
    if val_epoch_acc > best_:
      best_ = val_epoch_acc
      aug = 'aug' if params['augmentation'] else 'no_aug'
      torch.save(model, f'/home/usuaris/dduenas/scripts/models/{params["model"]}_{aug}.pth')
      print(f"Saving current best model: {best_:.3f}\n")
  except:
    pass

  if params['lr_scheduler']:
    lr_scheduler(val_epoch_loss)
  if params['early_stopping']:
    early_stopping(val_epoch_loss)
    if early_stopping.early_stop:
      break
  
  fit_lr.append(lr_)

  metrics = [(train_loss,val_loss), (train_acc,val_acc), (train_f1,val_f1), (train_auc,val_auc)]
  names = ['loss', 'accuracy', 'f1', 'auc']
  save_metrics(metrics, names, path=PATH, plot=False)
  save_epoch_lr(fit_lr, lr_label= f'learning rate', path=PATH, plot=False)
end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")