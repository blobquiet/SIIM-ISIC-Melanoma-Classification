import pandas as pd
import os

import numpy as np
from sklearn.model_selection import train_test_split

def to_path(image_name, path):
  return path + image_name + '.jpg'

def mapping(df):
  df['diagnosis'] = None
  df['diagnosis'].mask(df['MEL']==1, 'MEL', inplace=True)
  df['diagnosis'].mask(df['NV']==1, 'NV', inplace=True)
  df['diagnosis'].mask(df['BCC']==1, 'BCC', inplace=True)
  df['diagnosis'].mask(df['AK']==1, 'AK', inplace=True)
  df['diagnosis'].mask(df['BKL']==1, 'BKL', inplace=True)
  df['diagnosis'].mask(df['DF']==1, 'DF', inplace=True)
  df['diagnosis'].mask(df['VASC']==1, 'VASC', inplace=True)
  df['diagnosis'].mask(df['SCC']==1, 'SCC', inplace=True)
  df['diagnosis'].mask(df['UNK']==1, 'UNK', inplace=True)
  df['target'] = df['diagnosis']
  
  diagnosis_idx = {d: idx for idx, d in enumerate(sorted(df.diagnosis.unique()))}
  df['target'] = df['diagnosis'].map(diagnosis_idx)
  return df

def create_df(dir, file_name, export_csv=False, subset = False, whole = False):
  '''
  Creates a dataframe with the following assertions
  file_name: name of the csv file (if name is 'test' only a path is added)
  export_csv: export dataframe
  whole: return the whole dataset the path and mapping
  subset: return a subset of the dataset with path and mapping
  '''
  df = pd.read_csv(f'{dir}/{file_name}.csv')
  df['path'] = df.image.apply(to_path, path = f'{dir}/{file_name}/')  

  if file_name == 'test':
    if export_csv:
      df.to_csv(f'{dir}/{file_name}.csv', index=False)
    return df.reset_index(drop=True)
  
  mapping(df)
  if whole:
    if export_csv:
      df.to_csv(f'{dir}/{file_name}_whole.csv', index=False)
    return df.reset_index(drop=True)
  
  df, df_val = train_test_split(df, random_state=42, test_size=0.67, shuffle=True, stratify=df['target'])
  if subset:
    subset_, subset_val = train_test_split(df, random_state=42, test_size=0.33, shuffle=True, stratify=df['target'])
    if export_csv:
      df.to_csv(f'{dir}/subset.csv', index=False)
    return subset_.reset_index(drop=True), subset_val.reset_index(drop=True)
  if export_csv:
    df.to_csv(f'{dir}/train.csv', index=False)
    df_val.to_csv(f'{dir}/val.csv', index=False)
  return df.reset_index(drop=True), df_val.reset_index(drop=True)