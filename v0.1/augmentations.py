import albumentations as A

mean_= (0.66712415, 0.52883655, 0.52394277)
std_= (0.22350791, 0.20326485, 0.21510653)

mean_val = (0.66793597, 0.53040045, 0.5241613)
std_val = (0.22349131, 0.20395535, 0.21533556)


def transform_aug(im_size):
    transforms_train = A.Compose(
      [
       A.Transpose(p=0.5),
       A.VerticalFlip(p=0.5),
       A.HorizontalFlip(p=0.5),
       A.RandomBrightness(limit=0.2, p=0.75),
       A.RandomContrast(limit=0.2, p=0.75),
      #  A.OneOf([
      #           A.MotionBlur(blur_limit=5),
      #           A.MedianBlur(blur_limit=5),
      #           A.Blur(blur_limit=5),
      #           A.GaussNoise(var_limit=(5.0, 30.0)),
      #           ], p=0.7),
       A.OneOf([
                # A.OpticalDistortion(distort_limit=1.0),
                # A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
                ], p=0.7),
       A.CLAHE(clip_limit=4.0, p=0.7),
       A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
       A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=0, p=0.85),
       A.Resize(im_size, im_size),
       A.Cutout(max_h_size=int(im_size*0.175), max_w_size=int(im_size*0.175), num_holes=1, p=0.7),
       A.Normalize(mean_,std_)
       ])
  
    transforms_val = A.Compose(
      [
      #  A.SmallestMaxSize(max_size=160),
      #  A.CenterCrop(height=128, width=128),
       A.Resize(im_size, im_size),
       A.Normalize(mean_, std_)
       ])
    return transforms_train, transforms_val


def transform_normalize(im_size):
    transforms_train = A.Compose(
      [
       A.Resize(im_size, im_size),
       A.Normalize(mean_,std_)
       ])
    transforms_val = A.Compose(
      [
       A.Resize(im_size, im_size),
       A.Normalize(mean_val,std_val)
       ])
    return transforms_train, transforms_val