# TrainingCode -- Classification
The code is used as quick deploy for training classification task.

# Distribute Mode
The use of gpu can be setted in config.py DEVICE.  
```
python train_val.py
```

# Data Distribute Mode
The use of gpu can be setted in config.py DEVICE.  
```
python -m torch.distributed.launch --nproc_per_node <number_of_gpu> train_val.py
```