import torch

params = {
    #"model": "vit_large_patch16_224",
    "model": "resnet18",
    #"model": "inception_resnet_v2",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "im_size": 224,
    "lr": 3e-3,
    "batch_size": 64,
    "num_workers": 1,
    "epochs": 20,
    "folds":None,
    "focal_loss": True,
    "lr_scheduler": True,
    "early_stopping": True,
    "augmentation": True,
    "path": "/home/usuaris/dduenas/ISIC/2019"
}
PATH = params['path']

print("lr:",params['lr'])
print("batch:",params['batch_size'])
print("model:",params['model'])
print("augmentation:",params['augmentation'])
print("focal loss:",params['focal_loss'])