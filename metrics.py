from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch
from torch import nn
import matplotlib.pyplot as plt
threshold = 0.5

def auc_score(outputs, targets):
  # outputs = torch.sigmoid(outputs)
  outputs = outputs.cpu().detach().numpy()
  targets = targets.cpu().detach().numpy()
  auc = roc_auc_score(targets, outputs)
  return auc

def auc_score_m(outputs, targets, multi_class='ovr'):
  outputs = outputs.cpu()
  outputs = nn.functional.softmax(outputs, dim=1)
  outputs = outputs.detach().numpy()
  targets = targets.cpu().detach().numpy()
  auc = roc_auc_score(targets, outputs, multi_class=multi_class)
  return auc

def f1_score_(outputs, targets):
  outputs = outputs >= threshold
  # outputs = outputs >= 0.0
  targets = targets == 1.0
  outputs = outputs.cpu().detach().numpy()
  targets = targets.cpu().detach().numpy()
  f1 = f1_score(targets, outputs, average='weighted')
  return f1

def f1_score_m(outputs, targets):
  outputs = outputs.cpu()
  outputs = nn.functional.softmax(outputs, dim=1)
  _, outputs = torch.max(outputs, dim = 1)  
  outputs = outputs.detach().numpy()
  targets = targets.cpu().detach().numpy()
  f1 = f1_score(targets, outputs, average='weighted')
  return f1

def accuracy_score_m(outputs, targets):
    outputs = torch.log_softmax(outputs, dim = 1)
    _, outputs = torch.max(outputs, dim = 1)    
    preds = (outputs == targets).float()
    acc = preds.sum() / len(preds)
    return acc.cpu().detach().numpy()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_roc_curve_m(outputs, targets, path, label, pos_label, plot=False, multi_class='ovr'):
  outputs = outputs.cpu()
  outputs = nn.functional.softmax(outputs, dim=1)
  _, outputs_ = torch.max(outputs, dim = 1)  
  outputs = outputs.detach().numpy()
  targets = targets.cpu().detach().numpy()
  false_positive_rate, true_positive_rate, thresholds = roc_curve(targets, outputs_.detach().numpy(), pos_label=pos_label)
  auc = roc_auc_score(targets, outputs, multi_class='ovr')
  plt.plot(false_positive_rate[2],true_positive_rate[2],color="darkorange",label="ROC curve (area = %0.2f)" % auc)
  plt.plot(false_positive_rate, true_positive_rate, lw=2, color='darkorange')
  plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
  plt.xlim([-0.05, 1.05])
  plt.ylim([0.0, 1.05])
  plt.savefig(f"{path}/{label}.png")
  plt.legend(loc="lower right")
  plt.title('Receiver operating characteristic ROC')
  plt.show() if plot else plt.close()
  plt.clf()