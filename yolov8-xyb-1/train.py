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
When training your own object detection model, be sure to pay attention to the following points:
1. Before training, carefully check whether your format meets the requirements. This library requires the dataset format to be VOC format. The prepared content should include input images and labels.
   The input images are .jpg images and do not need to be a fixed size. They will be resized automatically before being fed into training.
   Grayscale images will be automatically converted to RGB images for training, no need to modify them yourself.
   If the suffix of the input image is not jpg, you need to batch convert them to jpg before starting training.

   The labels are in .xml format and the files contain information about the objects to be detected, corresponding to the input image files.

2. The size of the loss value is used to judge whether it is converging. What is important is the trend of convergence, i.e., the validation set loss keeps decreasing. If the validation set loss basically does not change, the model is basically converging.
   The specific size of the loss value has no significance. Large and small are only related to the way the loss is calculated. It is not better if it is close to 0. If you want the loss to look better, you can divide the corresponding loss function by 10000.
   The loss value during training will be saved in the logs folder under the loss_%Y_%m_%d_%H_%M_%S folder.

3. The trained weight files are saved in the logs folder. Each training epoch contains several training steps, and each training step performs a gradient descent.
   If only a few steps are trained, the weights will not be saved. It is necessary to clarify the concepts of Epoch and Step.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use Cuda
    #           If there is no GPU, set it to False
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    Used to fix the random seed
    #           So that the same results can be obtained for each independent training
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use single-machine multi-GPU distributed training
    #                   Terminal command only supports Ubuntu. CUDA_VISIBLE_DEVICES is used to specify GPUs in Ubuntu.
    #                   On Windows systems, DP mode is used to call all GPUs by default and does not support DDP.
    #   DP mode:
    #       Set            distributed = False
    #       In the terminal, enter    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       In the terminal, enter    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available in DDP mode with multiple GPUs
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Can reduce memory usage by about half, requires pytorch 1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    Point to the txt file under model_data, related to your training dataset
    #                   Before training, be sure to modify classes_path to correspond to your dataset
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/label-x.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   For weight file download, see README, you can download via network disk. Pre-trained weights of the model are general for different datasets because features are general.
    #   The important part of the pre-trained weights is the weight part of the backbone feature extraction network, which is used for feature extraction.
    #   Pre-trained weights must be used in 99% of cases. Without them, the weights of the backbone part are too random, and the feature extraction effect is not obvious. The network training result will not be good.
    #
    #   If there is an interruption during training, you can set model_path to the weight file under the logs folder and reload the weights that have been partially trained.
    #   At the same time, modify the parameters of the Freeze phase or Unfreeze phase below to ensure the continuity of the model epoch.
    #
    #   When model_path = '', the entire model weights are not loaded.
    #
    #   Here, the weights of the entire model are used, so they are loaded in train.py.
    #   If you want to train the model from scratch, set model_path = '', and below Freeze_Train = False, then start training from scratch without freezing the backbone.
    #
    #   Generally speaking, training the network from scratch is very poor because the weights are too random and the feature extraction effect is not obvious, so it is very, very, very not recommended to train from scratch!
    #   There are two solutions for training from scratch:
    #   1. Thanks to the powerful data augmentation ability of the Mosaic data augmentation method, set UnFreeze_Epoch to a larger value (300 or above), batch_size larger (16 or above), and more data (tens of thousands or above).
    #      You can set mosaic=True and start training with randomly initialized parameters, but the effect is still not as good as with pre-training. (Large datasets like COCO can do this)
    #   2. Understand the imagenet dataset, first train a classification model, obtain the weight of the backbone part of the network, and then train based on this.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolov8.pth'
    #------------------------------------------------------#
    #   input_shape     The size of the input shape must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [512, 512]
    #------------------------------------------------------#
    #   phi             The version of yolov8 to be used
    #                   n : Corresponds to yolov8_n
    #                   s : Corresponds to yolov8_s
    #                   m : Corresponds to yolov8_m
    #                   l : Corresponds to yolov8_l
    #                   x : Corresponds to yolov8_x
    #------------------------------------------------------#
    phi             = 's'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use the pre-trained weights of the backbone network, here it uses the backbone weights, so it is loaded during model construction.
    #                   If model_path is set, there is no need to load the backbone weights, the value of pretrained is meaningless.
    #                   If model_path is not set, pretrained = True, then only the backbone starts training.
    #                   If model_path is not set, pretrained = False, Freeze_Train = False, then start training from scratch without freezing the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    #   mosaic              Mosaic data augmentation.
    #   mosaic_prob         Probability of using mosaic data augmentation for each step, default is 50%.
    #
    #   mixup               Whether to use mixup data augmentation, only effective when mosaic=True.
    #                       Only the images enhanced by mosaic will be processed by mixup.
    #   mixup_prob          Probability of using mixup data augmentation after mosaic, default is 50%.
    #                       The total probability of mixup is mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio   Refer to YoloX, the training images generated by Mosaic are far from the real distribution of natural images.
    #                       When mosaic=True, this code will enable mosaic within the special_aug_ratio range.
    #                       Default is to enable mosaic for the first 70% of epochs, i.e., 70 out of 100 epochs.
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
    #   Training is divided into two phases: frozen phase and unfrozen phase. Setting the frozen phase is to meet the training needs of students with insufficient machine performance.
    #   The frozen training requires less memory. If the graphics card is very poor, you can set Freeze_Epoch equal to UnFreeze_Epoch and Freeze_Train = True, then only perform frozen training.
   

 #      
    #   Here are some parameter setting suggestions. Adjust flexibly according to your needs:
    #   (A) Start training from the pre-trained weights of the entire model: 
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (not frozen)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (not frozen)
    #       Among them, UnFreeze_Epoch can be adjusted between 100-300.
    #   (B) Start training from scratch:
    #       Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (not frozen training)
    #       Among them, UnFreeze_Epoch should be no less than 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    #   (C) Setting of batch_size:
    #       As large as possible within the acceptable range of the graphics card. Insufficient memory has nothing to do with the size of the dataset. If you encounter insufficient memory (OOM or CUDA out of memory), reduce the batch_size.
    #       Due to the impact of the BatchNorm layer, the batch_size should be at least 2 and cannot be 1.
    #       Normally, the Freeze_batch_size is recommended to be 1-2 times the Unfreeze_batch_size. It is not recommended to set the difference too large because it relates to the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Frozen phase training parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change.
    #   Requires less memory, only fine-tuning the network.
    #   Init_Epoch          The starting epoch of the model training, its value can be greater than Freeze_Epoch, such as setting:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #                       It will skip the frozen phase, start directly from 60 epochs, and adjust the corresponding learning rate.
    #                       (Used when training is interrupted and resumed)
    #   Freeze_Epoch        The epoch of frozen training for the model
    #                       (Invalid when Freeze_Train=False)
    #   Freeze_batch_size   The batch_size of frozen training for the model
    #                       (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 200
    Freeze_batch_size   = 4
    #------------------------------------------------------------------#
    #   Unfrozen phase training parameters
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change.
    #   Requires more memory, all parameters of the network will change.
    #   UnFreeze_Epoch          The total number of training epochs for the model
    #                           SGD requires longer time to converge, so a larger UnFreeze_Epoch is set
    #                           Adam can use relatively smaller UnFreeze_Epoch
    #   Unfreeze_batch_size     The batch_size of the model after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform frozen training
    #                   By default, the backbone is frozen first and then unfrozen for training.
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay related
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #   Min_lr          The minimum learning rate of the model, default is 0.01 times the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Types of optimizers used, optional are adam, sgd
    #                   When using the Adam optimizer, it is recommended to set Init_lr=1e-3
    #                   When using the SGD optimizer, it is recommended to set Init_lr=1e-2
    #   momentum        The momentum parameter used by the optimizer
    #   weight_decay    Weight decay, can prevent overfitting
    #                   adam will cause weight_decay error, it is recommended to set it to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   Types of learning rate decay methods used, optional are step, cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     Save weights every few epochs
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        Folder for saving weights and log files
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether to perform evaluation during training, the evaluation target is the validation set
    #                   After installing the pycocotools library, the evaluation experience is better.
    #   eval_period     Evaluate every few epochs, it is not recommended to evaluate frequently
    #                   Evaluation takes a lot of time, frequent evaluation will make training very slow
    #   The mAP obtained here will be different from the mAP obtained by get_map.py, there are two reasons:
    #   (1) The mAP obtained here is the mAP of the validation set.
    #   (2) The evaluation parameters set here are relatively conservative, the purpose is to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threading to read data
    #                   Enabling will speed up data reading, but will take up more memory
    #                   Computers with small memory can be set to 2 or 0  
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   train_annotation_path   Path to training images and labels
    #   val_annotation_path     Path to validation images and labels
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
    #   Get classes and anchors
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
        #   For weight file, see README, download from network disk
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load according to the Key of the pre-trained weights and the Key of the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key

.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Show Keys that did not match
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: It is normal that the head part is not loaded, but it is an error if the Backbone part is not loaded.\033[0m")

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
    #   torch 1.2 does not support amp, it is recommended to use torch 1.7.1 and above to correctly use fp16
    #   Therefore, "could not be resolve" is displayed in torch1.2 here
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-GPU SyncBatchNorm
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
        #   Total training epochs refer to the total number of times the entire dataset is traversed
        #   Total training steps refer to the total number of gradient descents 
        #   Each training epoch contains several training steps, and each training step performs a gradient descent.
        #   Here, only the minimum training epochs are recommended, no upper limit, and only the unfrozen part is considered in the calculation
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small to train, please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using %s optimizer, it is recommended to set the total training steps to at least %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data for this run is %d, Unfreeze_batch_size is %d, training for a total of %d Epochs, calculated total training steps is %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training steps are %d, less than the recommended total steps %d, it is recommended to set the total epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   The backbone feature extraction network features are general, frozen training can speed up training
    #   It can also prevent weights from being destroyed at the initial stage of training.
    #   Init_Epoch is the starting epoch
    #   Freeze_Epoch is the epoch of frozen training
    #   UnFreeze_Epoch total training epochs
    #   If OOM or insufficient memory is prompted, reduce the Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Partially frozen training
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If not freezing training, directly set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size, adaptively adjust the learning rate
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Select optimizer based on optimizer_type
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
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset

, shuffle=False,)
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
        #   Record eval mAP curve
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
            #   If the model has a frozen learning part
            #   Unfreeze it and set the parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size, adaptively adjust the learning rate
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