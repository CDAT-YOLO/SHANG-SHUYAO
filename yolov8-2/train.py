
#-------------------------------------#
#       Training on a dataset
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Pay attention to the following points when training your own object detection model:
1. Before training, carefully check whether your format meets the requirements. This library requires the dataset format to be VOC format. The prepared contents should include input images and labels.
   Input images should be .jpg images, without a fixed size, as they will be automatically resized before training.
   Grayscale images will be automatically converted to RGB images for training, no need to modify them yourself.
   If the suffix of the input images is not jpg, you need to batch convert them to jpg before starting training.

   Labels should be in .xml format, and the file should contain the target information to be detected. The label file should correspond to the input image file.

2. The size of the loss value is used to determine whether it has converged. The more important thing is the trend of convergence, i.e., the loss of the validation set decreases continuously. If the loss of the validation set does not change much, the model is basically converged.
   The specific size of the loss value does not have much significance. The size depends on the calculation method of the loss and does not necessarily need to be close to 0 to be good. If you want the loss to look better, you can directly divide by 10000 in the corresponding loss function.
   The loss value during the training process will be saved in the logs folder under the loss_%Y_%m_%d_%H_%M_%S folder.
   
3. The trained weight files are saved in the logs folder. Each training epoch contains several training steps, and each training step performs a gradient descent.
   If only a few steps are trained, it will not be saved. You need to clarify the concept of Epoch and Step.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use Cuda
    #           Set to False if no GPU is available
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    Used to set the random seed
    #           Ensures consistent results each time training is run independently
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use single-machine multi-card distributed operation
    #                   Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the GPU on Ubuntu.
    #                   On Windows systems, DP mode is used by default to call all GPUs, and DDP is not supported.
    #   DP mode:
    #       Set            distributed = False
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available in multi-card DDP mode
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Can reduce memory usage by about half, requires pytorch1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    Path to the txt in model_data, related to your training dataset
    #                   Be sure to modify classes_path before training to match your dataset
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   For downloading weight files, see README. They can be downloaded through cloud storage. Pretrained weights for models are universal for different datasets because features are universal.
    #   The important part of pretrained weights is the weights of the backbone feature extraction network, used for feature extraction.
    #   Pretrained weights must be used in 99% of cases. Without them, the weights of the backbone part are too random, the feature extraction effect is not obvious, and the training result of the network will not be good.
    #
    #   If there is an interruption during training, you can set model_path to the weight file in the logs folder and reload the weights that have already been trained.
    #   Also modify the parameters of the Freeze phase or Unfreeze phase below to ensure the continuity of the model epoch.
    #   
    #   When model_path = '', the weights of the entire model are not loaded.
    #
    #   Here, the weights of the entire model are used, so they are loaded in train.py.
    #   If you want to train the model from scratch, set model_path = '', and set Freeze_Train = False below. In this case, training starts from scratch, and there is no freezing process.
    #   
    #   Generally speaking, the training effect of the network from scratch will be very poor because the weights are too random, the feature extraction effect is not obvious. Therefore, it is highly recommended not to train from scratch!
    #   There are two options for training from scratch:
    #   1. Thanks to the strong data augmentation ability of the Mosaic data augmentation method, in the case of a large UnFreeze_Epoch (300 or above), large batch size (16 or above), and large data volume (10,000 or above),
    #      you can set mosaic=True and start training directly by randomly initializing the parameters, but the effect obtained is still not as good as with pre-training. (Large datasets like COCO can do this)
    #   2. Understand the imagenet dataset, first train the classification model to obtain the weights of the backbone part of the network. The backbone part of the classification model is universal with this model and then train based on this.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'logs/best_epoch_weights.pth'
    #------------------------------------------------------#
    #   input_shape     The size of the input shape, must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [512, 512]
    #------------------------------------------------------#
    #   phi             The version of yolov8 to use
    #                   n : Corresponds to yolov8_n
    #                   s : Corresponds to yolov8_s
    #                   m : Corresponds to yolov8_m
    #                   l : Corresponds to yolov8_l
    #                   x : Corresponds to yolov8_x
    #------------------------------------------------------#
    phi             = 's'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use pretrained weights for the backbone network. Here, the backbone weights are used and loaded when constructing the model.
    #                   If model_path is set, the backbone weights do not need to be loaded, and the value of pretrained is meaningless.
    #                   If model_path is not set, pretrained = True, in which case only the backbone is loaded to start training.
    #                   If model_path is not set, pretrained = False, and Freeze_Train = False, training starts from scratch, and there is no freezing process.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    #   mosaic              Mosaic data augmentation.
    #   mosaic_prob         The probability of using mosaic data augmentation for each step, default is 50%.
    #
    #   mixup               Whether to use mixup data augmentation, only effective when mosaic=True.
    #                       Only mixup processing is performed on images enhanced by mosaic.
    #   mixup_prob          The probability of using mixup data augmentation after mosaic, default is 50%.
    #                       The total mixup probability is mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio   Refer to YoloX. The training images generated by Mosaic are far from the real distribution of natural images.
    #                       When mosaic=True, this code will enable mosaic within the special_aug_ratio range.
    #                       By default, mosaic is enabled for the first 70% of epochs. For 100 epochs, 70 epochs will have mosaic enabled.
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   label_smoothing     Label smoothing. Generally below 0.01, such as 0.01 or 0.005.
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two phases: freezing phase and unfreezing phase. The freezing phase is set to meet the training needs of students with insufficient machine performance.
    #   Freezing training requires less memory, so if the GPU is very poor, set Freeze_Epoch equal to UnFreeze_Epoch and Freeze_Train = True,

 in which case only freezing training is performed.
    #      
    #   Here are some parameter setting suggestions for flexible adjustment based on your own needs:
    #   (a) Starting training from the pretrained weights of the entire model: 
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (Freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (Not freezing)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (Freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (Not freezing)
    #       Among them, UnFreeze_Epoch can be adjusted between 100-300.
    #   (b) Training from scratch:
    #       Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (Not freezing training)
    #       Among them, UnFreeze_Epoch should be at least 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    #   (c) Setting the batch size:
    #       Set the batch size as large as possible within the range acceptable to the GPU. Memory shortage has nothing to do with the size of the dataset. If there is a memory shortage (OOM or CUDA out of memory), reduce the batch size.
    #       Due to the influence of the BatchNorm layer, the batch size should be at least 2 and cannot be 1.
    #       Normally, the Freeze_batch_size is recommended to be 1-2 times that of the Unfreeze_batch_size. It is not recommended to set the difference too large because it is related to the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freezing phase training parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   Takes up less memory and only fine-tunes the network
    #   Init_Epoch          The current start epoch of the model, which can be greater than Freeze_Epoch, such as setting:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #                       This will skip the freezing phase, start directly from epoch 60, and adjust the corresponding learning rate.
    #                       (Used when resuming training from a checkpoint)
    #   Freeze_Epoch        The epoch for freezing training of the model
    #                       (Invalid when Freeze_Train=False)
    #   Freeze_batch_size   The batch size for freezing training of the model
    #                       (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    #   Unfreezing phase training parameters
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change
    #   Takes up more memory, and all parameters of the network will change
    #   UnFreeze_Epoch          The total training epoch of the model
    #                           SGD requires a longer time to converge, so a larger UnFreeze_Epoch is set
    #                           Adam can use a relatively smaller UnFreeze_Epoch
    #   Unfreeze_batch_size     The batch size of the model after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freezing training
    #                   By default, the backbone is frozen first and then unfrozen.
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay, etc.
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #   Min_lr          The minimum learning rate of the model, default is 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer used, options are adam, sgd
    #                   When using the Adam optimizer, it is recommended to set Init_lr=1e-3
    #                   When using the SGD optimizer, it is recommended to set Init_lr=1e-2
    #   momentum        The momentum parameter used inside the optimizer
    #   weight_decay    Weight decay to prevent overfitting
    #                   Adam will cause weight_decay errors, it is recommended to set to 0 when using Adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   The type of learning rate decay used, options are step, cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     How many epochs to save a weight
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        The folder where weights and log files are saved
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether to perform evaluation during training, the evaluation object is the validation set
    #                   After installing the pycocotools library, the evaluation experience is better.
    #   eval_period     Represents how many epochs to evaluate once, frequent evaluation is not recommended
    #                   Evaluation consumes a lot of time, frequent evaluation will cause training to be very slow
    #   The mAP obtained here will be different from that obtained in get_map.py for two reasons:
    #   (1) The mAP obtained here is the mAP of the validation set.
    #   (2) The evaluation parameters here are set conservatively to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threaded data reading
    #                   Enabling it will speed up data reading, but will take up more memory
    #                   Computers with less memory can set it to 2 or 0  
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   train_annotation_path   Training image paths and labels
    #   val_annotation_path     Validation image paths and labels
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    seed_everything(seed)
    #------------------------------------------------------#
    #   Set the GPU to use
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #------------------------------------------------------#
    #   Get classes and anchor
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #----------------------------------------------------#
    #   Download pretrained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)
            
    #------------------------------------------------------#
    #   Create yolo model
    #------------------------------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != '':
        #------------------------------------------------------#
        #   Weight files please see README, Baidu cloud download
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load based on the keys of the pretrained weights and the model's keys
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k]

 = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Display unmatched keys
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mKind reminder, it is normal that the head part is not loaded. If the Backbone part is not loaded, it is an error.\033[0m")

    #----------------------#
    #   Get the loss function
    #----------------------#
    yolo_loss = Loss(model)
    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2 does not support amp, it is recommended to use torch 1.7.1 and above to use fp16 correctly
    #   Therefore, torch1.2 shows "could not be resolved" here
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-card sync Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported on one GPU or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Parallel run on multiple cards
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            
    #----------------------------#
    #   Weight smoothing
    #----------------------------#
    ema = ModelEMA(model_train)
    
    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   Total training epoch refers to the total number of times the entire dataset is traversed
        #   Total training steps refer to the total number of gradient descents
        #   Each training epoch contains several training steps, and each training step performs a gradient descent.
        #   Here only the minimum training epoch is recommended, there is no upper limit, and only the unfreezing part is considered when calculating
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small to train, please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step to at least %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data volume for this run is %d, Unfreeze_batch_size is %d, a total of %d Epochs are trained, and the total training step is calculated to be %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training step is %d, which is less than the recommended total step of %d, it is recommended to set the total epoch to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   The backbone feature extraction network features are universal, and freezing training can speed up training
    #   It can also prevent the weights from being destroyed in the early stages of training.
    #   Init_Epoch is the starting epoch
    #   Freeze_Epoch is the epoch for freezing training
    #   UnFreeze_Epoch is the total training epoch
    #   If there is OOM or memory shortage, please reduce the Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze part of the training
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If not freezing training, directly set the batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size and adaptively adjust the learning rate
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Choose the optimizer based on optimizer_type
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #   Get the learning rate decay formula
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Determine the length of each epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, please expand the dataset.")

        if ema:
            ema.updates     = epoch_step * Init_Epoch
        
        #---------------------------------------#
        #   Construct dataset loader.
        #---------------------------------------#
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset

, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   Record the map curve of eval
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start training the model
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen learning part
            #   Unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size and adaptively adjust the learning rate
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get the learning rate decay formula
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, please expand the dataset.")
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if distributed:
                    batch_size  = batch_size // ngpus_per_node
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
