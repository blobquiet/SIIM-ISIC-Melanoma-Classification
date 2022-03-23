import matplotlib.pyplot as plt
import seaborn as sn
from params import *
from torch import nn
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

def save_metrics(metric_list, labels_list, path, plot=False):
  for i in range(len(labels_list)):
    plt.figure(figsize=(7, 4))
    plt.plot(metric_list[i][0], color='green', label=f'train {labels_list[i]}')
    plt.plot(metric_list[i][1], color='orange', label=f'validataion {labels_list[i]}')
    plt.xlabel('epochs')
    plt.ylabel(labels_list[i])
    plt.legend()
    plt.savefig(f"{path}/{labels_list[i]}.png")
    plt.show() if plot else plt.close()

def save_epoch_lr(lr_list, lr_label, path, plot = False):
  plt.figure(figsize=(7, 4))
  plt.plot(lr_list, color='red', label=lr_label)
  plt.xlabel('epochs')
  plt.ylabel(lr_label)
  plt.legend()
  plt.savefig(f"{path}/{lr_label}.png")
  plt.show() if plot else plt.close()
   
def save_confusion_matrix_m(outputs, targets, path, label, classes=None, plot=False, percentage=True):
  model_name= params['model']
  o = outputs.cpu()
  o = nn.functional.softmax(o, dim=1)
  _, o = torch.max(o, dim = 1)  
  o = o.detach().numpy()
  t = targets.cpu().detach().numpy()
  cf_matrix = confusion_matrix(t, o)
  df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) if percentage else cf_matrix, index = [i for i in classes], columns = [i for i in classes])
  sn.heatmap(df_cm, annot=True, cmap='Blues')
  plt.title(f'{label} Confusion Matrix')
  plt.savefig(f'{path}/{label}_cm.png')
  plt.show() if plot else plt.close()
  plt.clf()