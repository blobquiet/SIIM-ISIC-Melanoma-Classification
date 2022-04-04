
from torch.utils.data import DataLoader
from dataframe import create_df
from params import *
from dataset import MelanomaDataset

def get_mean_and_std(dataloader):
  '''
  recieves a torch dataloader and calculates the mean
  and standard deviation
  '''
  channels_sum, channels_squared_sum, num_batches = 0, 0, 0
  counter=0
  for data, _ in dataloader:
    counter+=1
    print("batch",counter)
    print("shape",data.shape)
    channels_sum += torch.mean(data, dim=[0,2,3])
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    num_batches += 1
  mean = channels_sum / num_batches
  std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
  return tuple(mean.numpy()), tuple(std.numpy())

df = create_df(dir = params['path'], file_name = 'train', whole=True)
# df['path'].to_frame()
dataset = MelanomaDataset(df['path'], labels=df['target'])
print(len(dataset))
dataset_loader = DataLoader(dataset, batch_size=len(df))
mean, std = get_mean_and_std(dataset_loader)
print(mean)
print(std)