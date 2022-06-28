from easydict import EasyDict as edict

cfg = edict()

# general
cfg.RANDOM_SEED = 1
cfg.BATCH_SIZE = 32
cfg.TEST_BATCH_SIZE = 1

cfg.VERBOSE = True
cfg.EPSILON = 1e-7
cfg.CLEANUP = False

cfg.DATASET = edict()
cfg.DATASET.SHUFFLE = True

cfg.MODEL = edict()

# define training params
cfg.TRAINING = edict()
cfg.TRAINING.OPTIMIZER = "adam"
cfg.TRAINING.START_LR = 0.0001
cfg.TRAINING.STEPS_PER_EPOCH = None


