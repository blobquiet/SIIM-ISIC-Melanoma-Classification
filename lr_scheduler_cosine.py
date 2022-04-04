from timm.scheduler.cosine_lr import CosineLRScheduler

class LRCosineScheduler():
  def __init__(self, optimizer, warmup_t=2, t_initial=20, warmup_lr_init=1e-3):
    self.optimizer = optimizer
    self.warmup_t = warmup_t
    self.t_initial = t_initial
    self.warmup_lr_init = warmup_lr_init
    self.lr_scheduler = CosineLRScheduler(self.optimizer, t_initial=self.t_initial,
    warmup_t=self.warmup_t, warmup_lr_init=self.warmup_lr_init)
  def __call__(self, val_loss):
    self.lr_scheduler.step(val_loss)