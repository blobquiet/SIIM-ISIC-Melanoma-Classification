import albumentations as A

mean= (0.66739434, 0.5293492, 0.52400774)
std= (0.22350878, 0.2035029, 0.21521305)

def transform_aug(im_size):
  transforms_train = A.Compose(
      [
       A.Transpose(p=0.5),
       A.VerticalFlip(p=0.5),
       A.HorizontalFlip(p=0.5),
       A.RandomBrightnessContrast(brightness_limit=0.2, p=0.75),
       A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.Blur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),
       A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
                ], p=0.7),
       A.CLAHE(clip_limit=4.0, p=0.7),
       A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
       A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, p=0.85),
       A.Resize(im_size, im_size),
       A.CoarseDropout(max_height=int(im_size * 0.175), max_width=int(im_size * 0.175), max_holes=1, p=0.7),
       A.Normalize(mean,std)
       ])
  transforms_val = A.Compose(
      [
       A.SmallestMaxSize(max_size=160),
       A.CenterCrop(height=128, width=128),
       A.Resize(im_size, im_size),
       A.Normalize(mean,std)
       ])
  return transforms_train, transforms_val
  
def transform_normalize(im_size):
  transforms_train = A.Compose(
      [
       A.Resize(im_size, im_size),
       A.Normalize(mean,std)
       ])
  transforms_val = A.Compose(
      [
       A.Resize(im_size, im_size),
       A.Normalize(mean,std)
       ])
  return transforms_train, transforms_val