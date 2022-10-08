import albumentations as A

mean= (0.66739434, 0.5293492, 0.52400774)
std= (0.22350878, 0.2035029, 0.21521305)

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