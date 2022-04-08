from statistics import mean
from metric_monitor import MetricMonitor
from metrics import auc_score_m, f1_score_m, accuracy_score_m, save_roc_curve_m
import torch
from visuals import save_confusion_matrix_m
from tqdm import tqdm
from params import *
from torch import nn
from focal_loss_ import *

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def fit(train_loader, model, criterion, optimizer, epoch, params, samples_per_cls=None):
  metric_monitor = MetricMonitor()
  model.train()
  stream = tqdm(train_loader)

  train_outputs = torch.tensor([]).to(params['device'])
  train_targets = torch.tensor([]).to(params['device'])
  train_loss = []

  val_steps=0

  for _, batch in enumerate(stream, start=1):
    images, targets = batch
    images = images.to(params["device"], non_blocking=True)
    targets = targets.to(params["device"], non_blocking=True).long().squeeze()
    outputs = model(images)
    if samples_per_cls!=None:
      loss =  CB_loss(outputs, targets, samples_per_cls, 8, 'focal', beta=0.9999, gamma=0.5)
    else:
      loss = criterion(outputs, targets)

    train_outputs = torch.cat([train_outputs, outputs])
    train_targets = torch.cat([train_targets.long(), targets])
    train_loss.append(loss.item())
    
    accuracy = accuracy_score_m(outputs, targets)
    f1 = f1_score_m(outputs, targets)
    metric_monitor.update("Loss", loss.item())
    metric_monitor.update("Accuracy", accuracy)
    metric_monitor.update("F1", f1)
        
    try:
      auc = auc_score_m(outputs, targets)
      metric_monitor.update("AUC", auc)
    except:
      pass

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #if val_steps==2:
    #  break
    #val_steps+=1
    stream.set_description("Epoch: {epoch}. Train. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
  accuracy = accuracy_score_m(train_outputs, train_targets)
  f1 = f1_score_m(train_outputs, train_targets)
  try:
    auc = auc_score_m(train_outputs, train_targets, multi_class='ovr')
  except:
    auc = None
  return mean(train_loss), accuracy, f1, auc


def validate(val_loader, model, criterion, epoch, params, samples_per_cls=None):
  metric_monitor = MetricMonitor()
  model.eval()
  stream = tqdm(val_loader)
  val_outputs = torch.tensor([]).to(params['device'])
  val_targets = torch.tensor([]).to(params['device'])
  val_loss = []
  val_steps = 0
  with torch.no_grad():
    for i, batch in enumerate(stream, start=1):
      images, target = batch
      images = images.to(params["device"], non_blocking=True)
      targets = target.to(params["device"], non_blocking=True).long().squeeze()
      outputs = model(images)
      if samples_per_cls!=None:
        loss =  CB_loss(outputs, targets, samples_per_cls, 8, 'focal', beta=0.9999, gamma=0.5)
      else:
        loss = criterion(outputs, targets)

      val_outputs = torch.cat([val_outputs, outputs])
      val_targets = torch.cat([val_targets.long(), targets])
      val_loss.append(loss.item())
      
      accuracy = accuracy_score_m(outputs, targets)
      f1 = f1_score_m(outputs, targets)
      metric_monitor.update("Loss", loss.item())
      metric_monitor.update("Accuracy", accuracy)
      metric_monitor.update("F1", f1)
      try:
        auc = auc_score_m(outputs, targets, multi_class='ovr')
        metric_monitor.update("AUC", auc)
      except:
        pass
      #if val_steps==2:
      #  break
      #val_steps+=1
      stream.set_description("Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
  accuracy = accuracy_score_m(val_outputs, val_targets)
  f1 = f1_score_m(val_outputs, val_targets)    
  try:
    auc = auc_score_m(val_outputs, val_targets, multi_class='ovr')
  except:
    auc = None

  try:  
    classes = ('NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK')
    model_name= params['model']
    save_confusion_matrix_m(val_outputs, val_targets, path=PATH, label=model_name, classes=classes, plot = False, percentage = True)
    
    save_roc_curve_m(val_outputs, val_targets, path=PATH, label=f'ROC {model_name}', plot=False, pos_label=1)

    o = val_outputs.cpu()
    o = nn.functional.softmax(o, dim=1)
    _, o = torch.max(o, dim = 1)  
    o = o.detach().numpy()
    t = val_targets.cpu().detach().numpy()
    plot_labels = ['NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK']

    report = classification_report(o, t, target_names=plot_labels)
    print(report)
    
  except:
    pass
  return mean(val_loss), accuracy, f1, auc