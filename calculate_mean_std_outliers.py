from tqdm import tqdm
import cv2
import pandas as pd

df_train = pd.read_csv(f'../data/meta_csv/train_whole_meta_stratified.csv').reset_index(drop=True)
df_val = pd.read_csv(f'../data/meta_csv/val_whole_meta_stratified.csv').reset_index(drop=True)

df = pd.concat([df_train, df_val], ignore_index=True)


for idx in tqdm(df.index):
    img_name = df.loc[idx,'path']
    img = cv2.cvtColor(cv2.imread(df.path[idx]), cv2.COLOR_BGR2RGB)    
    w,h,_ = img.shape
    df.loc[idx,'width'] = w
    df.loc[idx,'height'] = h
    df.loc[idx,'ratio'] = w/h

    df.loc[idx,'mean'] = img.mean()
    df.loc[idx,'std'] = img.std()
    
    img_norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    df.loc[idx,'mean_norm'] = img_norm.mean()
    df.loc[idx,'std_norm'] = img_norm.std()

df.to_csv(f'./whole_mean_std.csv', index=False)