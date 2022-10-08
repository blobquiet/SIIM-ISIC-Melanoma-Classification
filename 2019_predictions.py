# -*- coding: utf-8 -*-
"""isic pytorch lightning python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IIIehXNn3i0DsLxB3TzBSzEh-jMMoIqz

# Colab
"""


import albumentations
albumentations.__version__

import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision
import os
from pathlib import Path
import math
import cv2 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize

import torchvision
import torchvision.transforms as transforms
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
import albumentations as A
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torch
from torchvision import transforms
import timm
from torchmetrics import Accuracy, F1Score, Specificity, Precision, Recall, AUROC
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore') 
import timm.optim
import wandb
import torch.optim.lr_scheduler

"""# Meta csv"""
import pandas as pd
t = pd.read_csv("../data/meta_csv/train_whole_folds.csv")
t.fold.value_counts()

meta_train = pd.read_csv("../data/meta_csv/whole_data_meta_no_duplicates.csv")
meta_test = pd.read_csv("../data/meta_csv/test_path.csv")

# fix column name
meta_train['anatom_site_general'] = meta_train['anatom_site_general_challenge']
meta_train.drop(columns=['anatom_site_general_challenge', 'temporal_image'], inplace=True)

meta_train.info()

meta_test.info()

meta_train

nans_meta_train = meta_train.isna().sum()
nans_meta_test = meta_test.isna().sum()
print(nans_meta_train)
print("\n")
print(nans_meta_test)

"""## Fill nans"""

meta_train['sex'] = meta_train['sex'].fillna('unknown')
meta_train['sex'].value_counts()

meta_test['sex'] = meta_test['sex'].fillna('unknown')
meta_test['sex'].value_counts()

meta_train.age_approx= meta_train.age_approx.fillna(0.0).astype(int)
meta_train = meta_train.astype({"age_approx": 'str'})

meta_train['age_approx'] = meta_train['age_approx'].fillna('unknown')
print("Missing values? -> ",meta_train['age_approx'].isna().any())
meta_train['age_approx'].value_counts()

meta_test.age_approx= meta_test.age_approx.fillna(0.0).astype(int)
meta_test = meta_test.astype({"age_approx": 'str'})

meta_test['age_approx'] = meta_test['age_approx'].fillna('unknown')
print("Missing values? -> ",meta_test['age_approx'].isna().any())
meta_test['age_approx'].value_counts()

meta_train['anatom_site_general'] = meta_train['anatom_site_general'].fillna('unknown')
print("Missing values? -> ",meta_train['anatom_site_general'].isna().any())
meta_train['anatom_site_general'].value_counts()

meta_test['anatom_site_general'] = meta_test['anatom_site_general'].fillna('unknown')
print("Missing values? -> ",meta_test['anatom_site_general'].isna().any())
meta_test['anatom_site_general'].value_counts()

"""## Encode Metadata"""

meta_train

from sklearn.preprocessing import OneHotEncoder
hot_enconder = OneHotEncoder()

train_columns = meta_train.columns[3], meta_train.columns[4], meta_train.columns[6]
train_columns =list(train_columns)
hot_enconder.fit(meta_train[train_columns])
hot_enconder.categories_

hot_enconder.transform([['10', 'female','upper extremity']]).toarray()

"""# Dataset Meta"""

from PIL import Image
class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, meta, labels, trans=None, augmix = None):
        self.imgs = imgs
        self.labels = labels
        self.trans = trans
        self.augmix = augmix
        meta = hot_enconder.transform(meta).toarray()
        self.meta = meta

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        meta = torch.tensor(self.meta[ix])
        if self.augmix is not None:
          img = Image.open(self.imgs[ix])
          img = self.trans(img)
        else:
          img = cv2.imread(self.imgs[ix])
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          if self.trans and self.augmix is None:
            img = self.trans(image=img)['image']
            img = torch.tensor(img, dtype=torch.float).permute(2,0,1)
        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, meta.float(), label

class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            path='2019',
            file='split',
            subset=0,
            batch_size=32,
            train_trans=None,
            val_trans=None,
            num_workers=4,
            pin_memory=True,
            val_size=0.1,
            augmix = None,
            fold = None,
            **kwargs):
        super().__init__()
        self.path = path
        self.file = file
        self.train_trans = train_trans
        self.subset = subset
        self.val_trans = val_trans
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_size = val_size
        self.augmix = augmix
        self.t_df = None
        self.fold = fold

    def setup(self, stage=None):
        train = pd.read_csv(f'{self.path}/train{self.file}.csv')
        train = train.astype({"age_approx": 'str'})
        if self.fold is not None:
          train_ = train[train['fold'] != self.fold].reset_index(drop=True)
          val = train[train['fold'] != self.fold].reset_index(drop=True)
          train = train_
          print(f'Fold {self.fold} is validating')
        else:
          self.t_df = train
          val = pd.read_csv(f'{self.path}/val{self.file}.csv')
          val = val.astype({"age_approx": 'str'})
          print("Training samples: ", len(train))
          print("Validation samples: ", len(val))
        if self.subset:
            _, train = train_test_split(
                train,
                test_size=self.subset,
                shuffle=True,
                stratify=train['target'],
                random_state=42
            )
            print("Training only on", len(train), "samples")
    
        meta_columns = ['age_approx', 'sex', 'anatom_site_general']
        # train dataset
        self.train_ds = Dataset(
            train['path'].values,
            train[meta_columns],
            train['target'].values,
            trans = self.train_trans if self.train_trans else None,
            augmix = self.augmix if self.augmix is not None else None)
        # val dataset
        self.val_ds=Dataset(
            val['path'].values,
            val[meta_columns],
            val['target'].values,
            trans = self.val_trans if self.val_trans else None
        )

    def train_dataloader(self):
        # return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, sampler=ImbalancedDatasetSampler(self.train_ds, labels=self.t_df['target'].values), pin_memory=self.pin_memory)
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)


"""# Model Meta"""


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters(config)
        num_class = 8
        class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        if config['external_data']:
          num_class=9
          class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']
        self.num_class=num_class
        self.class_names=class_names
        # self.balanced_accuracy = Accuracy(num_classes=num_class, average='macro')
        self.f1 = F1Score(num_classes=num_class, average='macro')
        self.specificity = Specificity(num_classes=num_class, average='macro')
        self.precision_ = Precision(num_classes=num_class, average='macro')
        self.recall = Recall(num_classes=num_class, average='macro')
        self.auc = AUROC(num_classes=num_class)
        # self.focal_loss = FocalLoss(num_class)
        self.learning_rate= self.hparams.lr
        

    def validation_step(self, batch, batch_idx):
        x, m, y = batch
        y_hat = self(x, m)
        # if self.hparams.loss_weight:
        # val_loss = F.cross_entropy(y_hat, y, weight=self.hparams.class_weights)
        # else:
        val_loss = F.cross_entropy(y_hat, y)
        # val_loss = self.focal_loss(y_hat, y)
        # val_loss =  CB_loss(y_hat, y, [867, 3404, 2762, 276, 4914, 13832, 628, 5990, 282], 9, 'focal', beta=0.9999, gamma=0.5)
        # sch = self.lr_schedulers()
        # sch.step(val_loss)

        val_acc = accuracy(y_hat, y)
        val_bacc = balanced_accuracy_score(y.detach().cpu().numpy(), torch.argmax(y_hat.detach().cpu(), dim=1).numpy())
        val_f1 = self.f1(y_hat, y)
        val_specificity = self.specificity(y_hat, y)
        val_precision = self.precision_(y_hat, y)
        val_recall = self.recall(y_hat, y)
        val_auc = self.auc(y_hat, y)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_bacc', val_bacc, prog_bar=True)
        self.log('val_f1', val_f1)
        self.log('val_specificity', val_specificity)
        self.log('val_precision', val_precision)
        self.log('val_recall', val_recall)
        self.log('val_auc', val_auc)

        return {"y_hat": y_hat.detach().cpu(), "y": y.detach().cpu()}

    def configure_optimizers(self):
        if 'SAM' == self.hparams.optimizer:
          print('SAM')
          optimizer = SAM(self.parameters(), timm.optim.AdamP, lr=self.hparams.lr)
        elif 'AdamP' == self.hparams.optimizer:
          print('AdamP')
          optimizer = getattr(timm.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)  
        elif 'SGD' == self.hparams.optimizer:
          print('SGD')
          optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay =5e-04)
        else:
          print('Else')
          optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if 'scheduler' in self.hparams:
            scheduler = [(scheduler, params) for scheduler, params in self.hparams.scheduler.items()]
  
    
            scheduler = getattr(torch.optim.lr_scheduler, scheduler[0][0])(optimizer, **scheduler[0][1])
            return [optimizer], [scheduler]
        return optimizer

# list(df.target.value_counts().sort_index())

class ViT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = timm.create_model(self.hparams.backbone,pretrained=self.hparams.pretrained, num_classes=9)
    
    # def forward(self, x):
    #     return self.model(x)
    def forward(self, x, meta):        
        x = self.backbone(x)
        # get meta
        meta = self.meta_model(meta)
        # concatenate both
        z = torch.cat([x,meta],1)
        # pass to backbone
        return self.head(z)

    def compute_loss(self, batch):
        x, y = batch
        return F.cross_entropy(self(x), y, weight=self.hparams.class_weights)

    def training_step(self, batch, batch_idx):
        # optimizer = self.optimizers()
        # scheduler = self.lr_schedulers()
        x, y = batch
        y_hat = self(x)
        if self.hparams.optimizer == 'SAM':
          # first forward-backward pass
          loss = self.compute_loss(batch)
          self.manual_backward(loss)
          optimizer.first_step(zero_grad=True)
          # second forward-backward pass
          loss_2 = self.compute_loss(batch)
          self.manual_backward(loss_2)
          optimizer.second_step(zero_grad=True)
        else:
          # optimizer.zero_grad()
          # loss = self.compute_loss(batch)
          # loss = self.focal_loss(y_hat, y)
          loss = F.cross_entropy(y_hat, y, weight=self.hparams.class_weights)
          # loss =  CB_loss(y_hat, y, [867, 3404, 2762, 276, 4914, 13832, 628, 5990, 282], 9, 'focal', beta=0.9999, gamma=0.5)
          # self.manual_backward(loss)
          # optimizer.step()
        # step every N epochs
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
        #   scheduler.step()

        acc = accuracy(y_hat, y)
        # train_bacc = self.balanced_accuracy(y_hat, y)
        train_bacc = balanced_accuracy_score(y.detach().cpu().numpy(), torch.argmax(y_hat.detach().cpu(), dim=1).numpy())

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_bacc', train_bacc, prog_bar=True)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        all_yhat = torch.cat([x['y_hat'] for x in validation_step_outputs])
        all_y = torch.cat([x['y'] for x in validation_step_outputs])
        self.logger.experiment.log({"val_roc" : wandb.plot.roc_curve(all_y.numpy(), all_yhat.numpy(), self.class_names)})
        self.logger.experiment.log({"val_confusion_matrix" : wandb.plot.confusion_matrix(y_true=all_y.numpy(), preds=torch.argmax(all_yhat, dim=1).numpy(),class_names=self.class_names)})

class Model(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        # meta enconder
        n_features = timm.create_model(self.hparams.backbone,pretrained=self.hparams.pretrained, num_classes = self.num_class).get_classifier().in_features
        self.backbone = timm.create_model(self.hparams.backbone,pretrained=self.hparams.pretrained, num_classes = 0)

        self.meta_model = torch.nn.Sequential(
            torch.nn.Linear(32, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
        )
        # Backbone | head
        #(64x512 and 32x128)
        self.head = torch.nn.Linear(n_features+128, self.num_class, bias=True)
        

    def forward(self, x, meta):        
        x = self.backbone(x)
        # get meta
        meta = self.meta_model(meta)
        # concatenate both
        z = torch.cat([x,meta],1)
        # pass to backbone
        return self.head(z)

    def training_step(self, batch, batch_idx):
        x, m, y = batch
        y_hat = self(x,m)
        loss = F.cross_entropy(y_hat, y, weight=self.hparams.class_weights)
        acc = accuracy(y_hat, y)
        train_bacc = balanced_accuracy_score(y.detach().cpu().numpy(), torch.argmax(y_hat.detach().cpu(), dim=1).numpy())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_bacc', train_bacc, prog_bar=True)
        return loss
        
    def validation_epoch_end(self, validation_step_outputs):
        all_yhat = torch.cat([x['y_hat'] for x in validation_step_outputs])
        all_y = torch.cat([x['y'] for x in validation_step_outputs])
        self.logger.experiment.log({"val_roc" : wandb.plot.roc_curve(all_y.numpy(), all_yhat.numpy(), self.class_names)})
        self.logger.experiment.log({"val_confusion_matrix" : wandb.plot.confusion_matrix(y_true=all_y.numpy(), preds=torch.argmax(all_yhat, dim=1).numpy(),class_names=self.class_names)})

"""# Predict

## Test
"""

import pandas as pd
test_csv = "../data/meta_csv/test.csv"
test = pd.read_csv(test_csv)
test = test.astype({"age_approx": 'str'})
test.shape

"""Meta"""

import cv2
import torch
from PIL import Image
class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, meta, trans=None):
        self.imgs = imgs
        self.trans = trans
        meta = hot_enconder.transform(meta).toarray()
        self.meta = meta

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        meta = torch.tensor(self.meta[ix])
        img = cv2.imread(self.imgs[ix])        
        # img = center_crop(img, (700,700))
        # img = microscope_crop(img)
        # if test_outliers.path.isin([self.imgs[ix]]).any():
        #   img = microscope_crop(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = resize(img, 230, 230)
        # img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.trans:
            img = self.trans(image=img)['image']
        img = torch.tensor(img, dtype=torch.float).permute(2,0,1)
        return img, meta.float()


"""Meta"""

def predict(model, dl, tta = 1):
  model.eval()
  model.cuda()
  tta_preds = []
  for i in range(tta):
      preds = torch.tensor([]).cuda()
      with torch.no_grad():
          t = tqdm(dl)
          for b, x_ in enumerate(t):
              x = x_[0].cuda()
              m = x_[1].cuda()
              y_hat = model(x,m)
              preds = torch.cat([preds, y_hat])
      tta_preds.append(preds)
  tta_preds = torch.stack(tta_preds).mean(axis=0)
  return torch.softmax(tta_preds, axis=1).cpu().numpy()

# new
weights_loss = [37.77162629757785,  
 9.694493783303729,
 11.98243688254665,
 111.38775510204083,
 6.677814029363784,
 2.3896672504378285,
 52.146496815286625,
 5.496475327291037,
 116.12765957446808]
classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']

import wandb
from tqdm import tqdm
run = wandb.init()

models_ = [
           #'model-1c4t7va1:v2', # beit_large_patch16_512-512
           #'model-154fzg8r:v7', # SAM beit_large_patch16_512-512
           #'model-1sry6rzn:v2', # swinv2_large_window12to16_192to256_22kft1k-
           #'model-154fzg8r:v7', # deit
           'model-16po8ctw:v0', # volo_d5_512-512
           
]

sizes= [
        #512,
        #512,
        #256,
        #384,
        512
        ]

names = [
    '-model-heavy_aug_tta32-new_weight-test-meta'
    ]


for i,m in enumerate(models_):
  artifact = run.use_artifact(f'skin-lesson/skin-lession/{m}', type='model')
  artifact_dir = artifact.download()

  model = Model.load_from_checkpoint(checkpoint_path=f'./artifacts/{m}/model.ckpt')
  model.hparams

  im_size = sizes[i]
  trans_ = A.Compose([
        A.RandomResizedCrop(height = im_size, width = im_size,  scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation = cv2.INTER_CUBIC,p = 1),
        A.Rotate(p=0.5),
        A.Flip(p = 0.5),
        A.Affine(mode=4, p=0.5),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,  p=0.5),
        A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=15, val_shift_limit=20,p = 0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.0, 0.05), rotate_limit=0, interpolation=1, border_mode=0, p=0.5),
        A.OneOf([
                 A.Blur(blur_limit=5, p=0.3),
                 A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
                 A.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.4)
                 ],p=0.5),
  A.Normalize()
  ])
  
  meta_columns = ['age_approx', 'sex', 'anatom_site_general']
  dataset = Dataset(test['path'], test[meta_columns], trans_)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

  preds = predict(model, dataloader, tta=32)

  submission = pd.DataFrame({'image': test['image'].values,
                            'AK': preds[...,0],
                            'BCC': preds[...,1],
                            'BKL': preds[...,2],
                            'DF': preds[...,3],
                            'MEL': preds[...,4],
                            'NV': preds[...,5],
                            'SCC': preds[...,6],
                            'UNK': preds[...,7],
                            'VASC': preds[...,8]})
  
  model_name = '2019-'+model.hparams.backbone+ '-'+ m + names[i]
  submission.to_csv(f'../data/2019_prediction/submission_{model_name}.csv', index=False)
  preds_thresh_df = submission.copy()
  preds_thresh_df.loc[:, classes] *= weights_loss
  preds_thresh_df.loc[:, classes] = preds_thresh_df.loc[:, classes].div(preds_thresh_df.sum(axis=1), axis=0)
  model_name = model.hparams.backbone + 'thresholding-'+ m + names[i]
  preds_thresh_df.to_csv(f'../data/2019_prediction/threshold/submission_{model_name}.csv', index=False)

wandb.finish()