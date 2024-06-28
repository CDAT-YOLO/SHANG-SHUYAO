
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
Pay attention to the following points when training your own object detection model:
1. Before training, carefully check if your format meets the requirements. This library requires the dataset format to be VOC format. The prepared content includes input images and labels.
   Input images are .jpg images, no fixed size required, they will be resized automatically before being passed into training.
   Grayscale images will be automatically converted to RGB images for training, no need to modify them yourself.
   If the input image is not in jpg format, you need to batch convert them to jpg before starting training.

   Labels are in .xml format, the file contains the target information to be detected, and the label files correspond to the input image files.

2. The size of the loss value is used to determine whether it converges. It is important to have a convergence trend, i.e., the validation set loss keeps decreasing. If the validation set loss remains unchanged, the model is basically converged.
   The specific size of the loss value does not have any significance. Large and small only depend on the way the loss is calculated. It is not better if it is close to 0. If you want the loss to look better, you can divide it by 10000 in the corresponding loss function.
   The loss value during the training process will be saved in the loss_%Y_%m_%d_%H_%M_%S folder under the logs folder.
   
3. The trained weight files are saved in the logs folder. Each training epoch consists of several training steps, and each training step performs one gradient descent.
   If only a few steps are trained, it will not be saved. You need to understand the concepts of epoch and step clearly.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use Cuda
    #           Set to False if there is no GPU
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    Used to fix the random seed
    #           So that each independent training can get the same results
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use multi-GPU distributed training
    #                   Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the GPU in Ubuntu.
    #                   Windows system defaults to using DP mode to call all GPUs, and does not support DDP.
    #   DP mode:
    #       Set            distributed = False
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available in DDP mode with multiple GPUs
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Can reduce about half of the memory usage, requires pytorch 1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    Points to the txt file under model_data, related to your own dataset 
    #                   Be sure to modify classes_path before training to match your own dataset
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   For weight file download, see README. You can download through the cloud disk. The pre-trained weights of the model are common for different datasets because the features are common.
    #   The pre-trained weights are important for the backbone feature extraction network weights, used for feature extraction.
    #   Pre-trained weights must be used for 99% of the situations. Without them, the backbone weights are too random, the feature extraction effect is not obvious, and the network training results will not be good.
    #
    #   If there is an interruption during training, you can set model_path to the weight file in the logs folder to reload the weights that have been trained for a part.
    #   Also modify the parameters of the freeze phase or unfreeze phase below to ensure the continuity of the model epochs.
    #   
    #   When model_path = '', the weights of the entire model are not loaded.
    #
    #   The weights of the entire model are used here, so it is loaded in train.py.
    #   If you want to train the model from scratch, set model_path = '', and Freeze_Train = False below, then it will start training from scratch without freezing the backbone.
    #   
    #   Generally speaking, the effect of training from scratch is very poor because the weights are too random and the feature extraction effect is not obvious. Therefore, it is highly, highly, highly not recommended to train from scratch!
    #   There are two schemes for training from scratch:
    #   1. Thanks to the strong data augmentation capability of Mosaic data augmentation method, set UnFreeze_Epoch to be relatively large (300 or above), batch size to be large (16 or above), and data to be large (more than 10,000). 
    #      You can set mosaic=True to start training with randomly initialized parameters directly, but the effect is still not as good as with pre-training. (It can be done with a large dataset like COCO)
    #   2. Understand the imagenet dataset, first train the classification model to get the backbone part weights, the backbone part of the classification model is common with this model, and then train based on this.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ''
    #------------------------------------------------------#
    #   input_shape     The input shape size, must be a multiple of 32
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
    #   pretrained      Whether to use the pre-trained weights of the backbone network, the pre-trained weights of the backbone are loaded during model construction.
    #                   If model_path is set, the weights of the backbone do not need to be loaded, and the value of pretrained is meaningless.
    #                   If model_path is not set, pretrained = True, then only the backbone is loaded to start training.
    #                   If model_path is not set, pretrained = False, Freeze_Train = False, then it will start training from scratch without freezing the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    #   mosaic              Mosaic data augmentation.
    #   mosaic_prob         The probability of using mosaic data augmentation for each step, default is 50%.
    #
    #   mixup               Whether to use mixup data augmentation, only effective when mosaic=True.
    #                       Mixup processing is only applied to mosaic augmented images.
    #   mixup_prob          The probability of using mixup data augmentation after mosaic, default is 50%.
    #                       The total probability of mixup is mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio   Refers to YoloX. Because the training images generated by Mosaic deviate far from the true distribution of natural images,
    #                       when mosaic=True, this code will enable mosaic within the special_aug_ratio range.
    #                       Defaults to the first 70% of epochs, enabling mosaic for 70 out of 100 epochs.
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
    #   Training is divided into two phases, freeze phase and unfreeze phase. The freeze phase is set to meet the training needs of students with insufficient machine performance.
    #   Freezing training requires less memory, and in the case of very poor graphics cards, you can set Freeze_Epoch equal to UnFreeze_Epoch, Freeze_Train = True, in which case only freezing training is performed.
    #      
    #   Here are some parameter setting suggestions. Adjust flexibly according to your own needs:
    #

   (1) Training starts from the pre-trained weights of the entire model:
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (freeze)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (no freeze)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (freeze)
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (no freeze)
    #       Among them: UnFreeze_Epoch can be adjusted between 100-300.
    #   (2) Training from scratch:
    #       Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (no freeze training)
    #       Among them: UnFreeze_Epoch is preferably not less than 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    #   (3) Setting batch_size:
    #       Set as large as possible within the acceptable range of the graphics card. If the memory is insufficient, please reduce batch_size.
    #       Due to the impact of the BatchNorm layer, the minimum batch_size is 2, not 1.
    #       Under normal circumstances, Freeze_batch_size is recommended to be 1-2 times Unfreeze_batch_size. It is not recommended to set the difference too large, as it relates to the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Parameters for freeze phase training
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change.
    #   Occupies less memory, only fine-tuning the network
    #   Init_Epoch          The current starting epoch of the model, its value can be greater than Freeze_Epoch, such as setting:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #                       It will skip the freeze phase, start directly from epoch 60, and adjust the corresponding learning rate.
    #                       (Used for resuming training)
    #   Freeze_Epoch        The freeze training epoch of the model
    #                       (Invalid when Freeze_Train=False)
    #   Freeze_batch_size   The batch size during freeze training of the model
    #                       (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    #   Parameters for unfreeze phase training
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change
    #   Occupies more memory, all parameters of the network will change
    #   UnFreeze_Epoch          The total number of training epochs of the model
    #                           SGD requires longer convergence time, so set a larger UnFreeze_Epoch
    #                           Adam can use relatively small UnFreeze_Epoch
    #   Unfreeze_batch_size     The batch size after unfreezing the model
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freeze training
    #                   Default is to freeze the backbone first and then unfreeze for training.
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, related to learning rate decay
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #   Min_lr          The minimum learning rate of the model, default is 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer used, options are adam, sgd
    #                   When using Adam optimizer, it is recommended to set Init_lr=1e-3
    #                   When using SGD optimizer, it is recommended to set Init_lr=1e-2
    #   momentum        The momentum parameter used in the optimizer
    #   weight_decay    Weight decay to prevent overfitting
    #                   Adam will cause weight_decay error, it is recommended to set it to 0 when using Adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   The learning rate decay method used, options are step, cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     Save the weights every few epochs
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        The folder where the weights and log files are saved
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether to evaluate during training, the evaluation object is the validation set
    #                   The evaluation experience is better after installing the pycocotools library.
    #   eval_period     Represents the number of epochs to evaluate once, frequent evaluation is not recommended
    #                   Evaluation consumes a lot of time, frequent evaluation will cause very slow training
    #   The mAP obtained here will be different from that obtained by get_map.py for two reasons:
    #   (1) The mAP obtained here is the mAP of the validation set.
    #   (2) The evaluation parameters set here are relatively conservative, in order to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threaded data reading
    #                   Enabling it will speed up data reading, but will occupy more memory
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
    #   Create YOLO model
    #------------------------------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != '':
        #------------------------------------------------------#
        #   The weight files can be found in the README, download from Baidu Cloud Disk
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load weights based on the Key of pre-trained weights and model's Key
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
        #   Display the Keys that were not matched
        #------------------------------------------------------#
        if local_rank == 0:


            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mKind Reminder, it is normal if the head part is not loaded, it is an error if the Backbone part is not loaded.\033[0m")

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
    #   torch 1.2 does not support amp, it is recommended to use torch 1.7.1 or above to correctly use fp16
    #   Therefore, torch 1.2 here shows "could not be resolved"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-GPU Sync BatchNorm
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported with one GPU or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel run
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
    #   Read dataset corresponding txt
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
        #   Total training epochs refer to the total number of times the entire dataset is traversed
        #   Total training steps refer to the total number of gradient descents 
        #   Each training epoch contains several training steps, each training step performs one gradient descent.
        #   Here, only the minimum training epochs are suggested, there is no upper limit, and only the unfrozen part is considered when calculating
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small to train, please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using %s optimizer, it is recommended to set the total training steps to more than %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data for this run is %d, Unfreeze_batch_size is %d, a total of %d Epochs are trained, and the total training steps are calculated to be %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training steps are %d, which is less than the recommended total steps %d, it is recommended to set the total epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   The backbone feature extraction network features are general, freezing training can speed up training
    #   It can also prevent the weights from being destroyed in the early stages of training.
    #   Init_Epoch is the starting epoch
    #   Freeze_Epoch is the epoch for freezing training
    #   UnFreeze_Epoch is the total training epoch
    #   If OOM or out of memory occurs, please reduce the Batch_size
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
        #   If not freezing training, directly set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size and adjust the learning rate adaptively
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
        #   Build dataset loader.
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

        gen             =

 DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
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
            #   If the model has a part to freeze for training
            #   Then unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size and adjust the learning rate adaptively
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
