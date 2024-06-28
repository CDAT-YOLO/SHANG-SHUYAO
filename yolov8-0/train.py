#-------------------------------------#
#       Train the dataset
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
When training your own object detection model, pay attention to the following points:
1. Before training, carefully check if your format meets the requirements. This library requires the dataset to be in VOC format. The prepared content should include input images and labels.
   Input images are .jpg files, with no fixed size, and will be resized automatically before being fed into training.
   Grayscale images will be automatically converted to RGB images for training, no need for manual conversion.
   If the input images are not in jpg format, you need to batch convert them to jpg before starting training.

   Labels are in .xml format, containing the information of objects to be detected, corresponding to the input image files.

2. The size of the loss value is used to determine whether the training is converging. It is important to see a trend of convergence, i.e., the validation set loss continuously decreases. If the validation set loss does not change much, the model is likely converged.
   The specific value of the loss does not matter. It is only meaningful for comparison. Large or small values depend on the way the loss is calculated, not necessarily better if close to zero. To make the loss look better, you can divide the loss function by 10000.
   The loss value during training is saved in the loss_%Y_%m_%d_%H_%M_%S folder under the logs directory.

3. The trained weight files are saved in the logs folder. Each epoch contains several training steps, and each training step performs a gradient descent.
   If you only train a few steps, the weights will not be saved. Understand the concepts of epoch and step clearly.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use Cuda
    #           Set to False if no GPU
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    Used to fix random seed
    #           Ensures the same results in each training
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use single-machine multi-GPU distributed training
    #                   Terminal command only supports Ubuntu. CUDA_VISIBLE_DEVICES is used to specify GPUs in Ubuntu.
    #                   On Windows, it defaults to DP mode and calls all GPUs, DDP is not supported.
    #   DP mode:
    #       Set            distributed = False
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available for multi-GPU in DDP mode
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Can reduce about half of the memory usage, requires pytorch1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    Points to the txt under model_data, related to the dataset
    #                   Before training, make sure to modify classes_path to match your dataset
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   For weight file downloads, see README, you can download from a network disk. Pre-trained weights are universal for different datasets because features are universal.
    #   The important part of pre-trained weights is the backbone feature extraction network weights for feature extraction.
    #   Pre-trained weights must be used in 99% of cases. Without them, the backbone weights are too random, resulting in poor feature extraction and poor training results.
    #
    #   If there are interruptions during training, set model_path to the weight file in the logs folder to reload the partially trained weights.
    #   Modify the parameters of the frozen stage or the unfrozen stage below to ensure the continuity of model epochs.
    #   
    #   When model_path = '', do not load the entire model weights.
    #
    #   Here, the entire model weights are used, so they are loaded in train.py.
    #   To train the model from scratch, set model_path = '', Freeze_Train = False, starting from scratch without freezing the backbone.
    #   
    #   Generally, training from scratch yields poor results due to random weights and ineffective feature extraction, hence strongly not recommended!
    #   Two methods to train from scratch:
    #   1. Benefit from the strong data augmentation of the Mosaic method, set UnFreeze_Epoch larger (300+), batch size larger (16+), and data larger (tens of thousands).
    #      Set mosaic=True and start training with random initialization. The result is still not as good as with pre-training. (For large datasets like COCO)
    #   2. Understand the imagenet dataset, train a classification model first to obtain backbone weights, which are common with this model for further training.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ''
    #------------------------------------------------------#
    #   input_shape     Input shape size, must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [512, 512]
    #------------------------------------------------------#
    #   phi             The version of yolov8 used
    #                   n : corresponds to yolov8_n
    #                   s : corresponds to yolov8_s
    #                   m : corresponds to yolov8_m
    #                   l : corresponds to yolov8_l
    #                   x : corresponds to yolov8_x
    #------------------------------------------------------#
    phi             = 's'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use pre-trained weights of the backbone network, loaded during model construction.
    #                   If model_path is set, backbone weights need not be loaded, pretrained value is meaningless.
    #                   If model_path is not set, pretrained = True, only load the backbone for training.
    #                   If model_path is not set, pretrained = False, Freeze_Train = False, start training from scratch without freezing the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    #   mosaic              Mosaic data augmentation.
    #   mosaic_prob         Probability of using mosaic data augmentation per step, default is 50%.
    #
    #   mixup               Whether to use mixup data augmentation, effective only when mosaic=True.
    #                       Only applies mixup to images augmented with mosaic.
    #   mixup_prob          Probability of using mixup after mosaic data augmentation, default is 50%.
    #                       Total mixup probability is mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio   Referencing YoloX, mosaic-generated training images deviate significantly from the real distribution of natural images.
    #                       When mosaic=True, this code will enable mosaic within special_aug_ratio range.
    #                       Default is the first 70% of epochs, for 100 epochs, 70 epochs will enable mosaic.
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   label_smoothing     Label smoothing. Generally below 0.01, e.g., 0.01, 0.005.
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two stages: freezing stage and unfreezing stage. Setting the freezing stage is to meet the training needs of those with insufficient machine performance.
    #   Frozen training requires less memory, for very poor graphics cards, set Freeze_Epoch equal to UnFreeze_Epoch, Freeze_Train = True, for only frozen training.
    #      
    #   Here are some parameter setting suggestions for flexible adjustment according to your needs:
    #   (1) Training from pre-trained weights of the entire model: 
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (not frozen)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (not frozen)
    #       Among them, UnFreeze_Epoch can be adjusted between 100-300.
    #   (2) Training from scratch:
    #       Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (not frozen)
    #       Among them, UnFreeze_Epoch should not be less than 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    #   (3) Setting batch_size:
    #       The larger the better within the acceptable range of the GPU. If out of memory (OOM or CUDA out of memory), reduce batch_size.
    #       BatchNorm layers require a minimum batch_size of 2, cannot be 1.
    #       Normally, Freeze_batch_size should be 1-2 times Unfreeze_batch_size. Avoid setting too large a difference as it affects the learning rate adjustment.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Parameters for the frozen training stage
    #   At this point, the backbone is frozen, the feature extraction network does not change
    #   Requires less memory, only fine-tunes the network
    #   Init_Epoch          The current starting epoch, can be greater than Freeze_Epoch, e.g., set:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #                       This skips the frozen stage, starts from epoch 60, and adjusts the learning rate accordingly.
    #                       (used for resuming interrupted training)
    #   Freeze_Epoch        The epoch for frozen training
    #                       (ineffective if Freeze_Train=False)
    #   Freeze_batch_size   The batch size for frozen training
    #                       (ineffective if Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    #   Parameters for the unfrozen training stage
    #   At this point, the backbone is not frozen, the feature extraction network changes
    #   Requires more memory, all network parameters change
    #   UnFreeze_Epoch          The total training epochs
    #                           SGD requires longer to converge, hence larger UnFreeze_Epoch
    #                           Adam can use relatively smaller UnFreeze_Epoch
    #   Unfreeze_batch_size     The batch size for training after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform frozen training
    #                   Default is to freeze the backbone first, then train unfrozen.
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #   Min_lr          The minimum learning rate of the model, default is 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer used, can be adam or sgd
    #                   For Adam optimizer, suggest setting Init_lr=1e-3
    #                   For SGD optimizer, suggest setting Init_lr=1e-2
    #   momentum        The momentum parameter used in the optimizer
    #   weight_decay    Weight decay to prevent overfitting
    #                   Adam can cause weight_decay errors, suggest setting to 0 when using Adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   The type of learning rate decay, can be step or cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     How many epochs to save the weights once
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        The folder to save weights and log files
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether to evaluate during training, the evaluation target is the validation set
    #                   Install pycocotools library for better evaluation experience.
    #   eval_period     Indicates how many epochs to evaluate once, frequent evaluation is not recommended
    #                   Evaluation consumes a lot of time, frequent evaluation slows down training
    #   The mAP obtained here will be different from get_map.py, for two reasons:
    #   (1) The mAP obtained here is for the validation set.
    #   (2) The evaluation parameters here are more conservative to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threaded data loading
    #                   Enabling it speeds up data loading but uses more memory
    #                   Computers with small memory can set it to 2 or 0  
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
    #   Set the GPU to be used
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
    #   Download pre-trained weights
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
        #   Weight file see README, download from Baidu Cloud
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load according to the Key of the pre-trained weights and the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Show the keys that did not match
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mFriendly reminder, it is normal for the head part not to load, but an error if the Backbone part does not load.\033[0m")

    #----------------------#
    #   Get loss function
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
    #   torch 1.2 does not support amp, recommend using torch 1.7.1 or above for correct fp16 usage
    #   Therefore, torch1.2 here shows "could not be resolved"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-GPU sync Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported with one GPU or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel running
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            
    #----------------------------#
    #   Smooth weights
    #----------------------------#
    ema = ModelEMA(model_train)
    
    #---------------------------#
    #   Read the corresponding txt of the dataset
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
        #   Total training epochs refer to the total number of times all data is traversed
        #   Total training steps refer to the total number of gradient descents 
        #   Each training epoch contains several training steps, each training step performs a gradient descent.
        #   Here we only suggest the minimum training epochs, no upper limit, considering only the unfrozen part
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small to train, please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using %s optimizer, it is recommended to set the total training steps to more than %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] This run's total training data is %d, Unfreeze_batch_size is %d, training %d epochs, calculated total training steps are %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training steps are %d, less than the recommended total steps %d, it is recommended to set the total epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   The backbone feature extraction network's features are universal, frozen training speeds up training
    #   It also prevents weights from being destroyed in the initial training phase.
    #   Init_Epoch is the starting epoch
    #   Freeze_Epoch is the epoch for frozen training
    #   UnFreeze_Epoch is the total training epoch
    #   If OOM or insufficient memory, reduce Batch_size
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
        #   If not freezing, set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Adjust learning rate adaptively based on current batch_size
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Choose optimizer based on optimizer_type
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
        #   Build dataset loader
        #---------------------------------------#
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
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
        #   Record eval map curve
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen part
            #   Unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Adjust learning rate adaptively based on current batch_size
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
