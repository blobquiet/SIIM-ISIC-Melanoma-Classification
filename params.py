import torch

params = {
    # "model": "vit_large_patch16_224",
    "model": "resnet18",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "im_size": 224,
    "lr": 1e-3,
    "batch_size": 16,
    "num_workers": 2,
    "epochs": 20,
    "folds":None,
    "lr_scheduler": True,
    "early_stopping": True,
    "augmentation": False,
    "path": "/content/drive/MyDrive/ISIC/2019"
}
PATH = params['path']