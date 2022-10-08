from sklearn.utils import class_weight
from params import *
import numpy as np

def get_weights(target_df, num_class=8):
    y = target_df.values
    class_weights = class_weight.compute_class_weight(class_weight = "balanced",classes = np.unique(y),y = y)
    # class_weights = class_weights / sum(class_weights) * num_class
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(params['device'])
    return class_weights