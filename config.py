from easydict import EasyDict

cfg = EasyDict()

# Remark
cfg.INDEX = "MNIST"

cfg.INFO_SHOW = ["std", "log"]

# hyper paramter
cfg.EPOCH = 1
cfg.RESUME = False

### if train and validation is mix together
cfg.KFOLD = 1

### If KFOLD greater than 1, VAL_SPLIT will not work.
cfg.VAL_SPLIT = 0.3
cfg.LR = 1e-4
cfg.WD = 1e-4

# device
cfg.DEVICE = [0]

# model
cfg.MODEL = "ResNet18"
cfg.BASIC_MODEL = "ResNet18"

# weight path
cfg.PRETRAINED_PATH = "../pretrained/resnet18.pth"
# cfg.PARAMETER_PATH = "../pretrained/resnet18.pth"

# loss function
cfg.LOSS = "CE"

# optimizer
cfg.OPTIMIZER = "ADAM"

# learning rate schedule
cfg.LR_SCHEUDLER = 50
cfg.joint_lr_step_size = 5

# dataset path and dataloader
### package name
cfg.DATALOADER = "load_data_train_val_classify"
cfg.TRAIN_DATASET_PATH = "../datasets/MNIST/Original/train"
cfg.VAL_DATASET_PATH = "../datasets/MNIST/Original/test"
cfg.DATASET_NAME = "MNIST"
cfg.CATEGORY = 10

# hyper paramter
cfg.TRAIN_BATCH_SIZE = 128
cfg.TRAIN_NUMBER_WORKDERS = 8
# Not work when using load_data_mix_all
cfg.TRAIN_SHUFFLE = True

# preprocess
cfg.TRAIN_IMAGE_SIZE = 224
cfg.TRAIN_RANDOM_CROP_SIZE = 224
cfg.TRAIN_RANDOM_FILP = True
cfg.TRAIN_NORM = True
cfg.TRAIN_MEAN = [0.485, 0.456, 0.406]
cfg.TRAIN_STD = [0.229, 0.224, 0.225]


# hyper paramter
cfg.VAL_BATCH_SIZE = 128
cfg.VAL_NUMBER_WORKDERS = 8
### Not work when using load_data_mix_all
cfg.VAL_SHUFFLE = False

# preprocess
cfg.VAL_IMAGE_SIZE = 256
# cfg.VAL_RANDOM_CROP_SIZE = 224
cfg.VAL_CENTER_CROP_SIZE = 224
cfg.VAL_RANDOM_FILP = False
cfg.VAL_NORM = True
cfg.VAL_MEAN = [0.485, 0.456, 0.406]
cfg.VAL_STD = [0.229, 0.224, 0.225]

