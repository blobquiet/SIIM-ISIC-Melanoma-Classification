import os
import cv2
import numpy as np
import albumentations
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

class MelanomaDataset(Dataset):
  def __init__(self, images, labels=None, transform=None, train=True):
    self.images = images
    self.labels = labels
    self.transform = transform
    self.train = train

  def __getitem__(self, index):
    image = cv2.imread(self.images[index])
    if self.transform is not None:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = self.transform(image=image)["image"]
      image = image.transpose(2, 0, 1)
    else:
      image = cv2.resize(image, (224,224))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = image/ 255.
      image = image.transpose(2, 0, 1)
    if self.train:
      target = torch.tensor([self.labels[index]])
      return torch.tensor(image).float(), target.float()
    return torch.tensor(image).float()

  def __len__(self) -> int:
    return len(self.images)