import torch

params = {
    #"model": "vit_large_patch16_224",
    #"model": "resnet18",
    "model": "efficientnet_b4",
    #"model": "inception_resnet_v2",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "im_size": 380,
    "lr": 1e-3,
    "batch_size": 16,
    "num_workers": 8,
    "epochs": 30,
    "folds":None,
    "focal_loss": False,
    "cosine_scheduler": False,
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
print("cosine scheduler:",params['cosine_scheduler'])
print(params)