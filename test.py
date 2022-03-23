from params import *
from dataframe import create_df
from dataset import MelanomaDataset
from torch.utils.data import DataLoader
import pandas as pd


from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import time

from metric_monitor import MetricMonitor


test = create_df(dir=params['path'], file_name='train', whole = True)
test= test[:400]
test_dataset = MelanomaDataset(test['path'], train=False)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

start = time.time()

PATH = params['path']
aug = 'aug' if params['augmentation'] else 'no_aug'
model_name=params['model']

model = torch.load(f'{PATH}/{model_name}_{aug}.pth',map_location=torch.device(params['device']))
model.to(params['device'])
model.eval()
preds = torch.tensor([]).to(params['device'])
val_outputs = torch.tensor([])
val_targets = torch.tensor([])
stream = tqdm(test_loader)
metric_monitor = MetricMonitor()

with torch.no_grad():
  for i, batch in enumerate(stream, start=1):
    imgs = batch
    imgs = imgs.to(params['device'])
    outputs = model(imgs)
    print("preds",preds.shape)
    preds = torch.cat([preds, outputs.view(-1)])
    print("\n")
    val_outputs = torch.cat([val_outputs, outputs.cpu()])
print('INFO: Test completed')
submission = pd.DataFrame({'image_name': test['image'].values, 'target': preds.cpu().numpy()})
submission.to_csv(f'{PATH}/submission.csv', index=False)
end = time.time()
print(f"\nTest time: {(end-start)/60:.3f} minutes")