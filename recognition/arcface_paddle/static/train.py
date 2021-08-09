# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import sys
import numpy as np

import paddle
from visualdl import LogWriter

from utils.logging import AverageMeter, init_logging, CallBackLogging
from datasets import CommonDataset, SyntheticDataset
from utils import losses

from .utils.verification import CallBackVerification
from .utils.io import Checkpoint

from . import classifiers
from . import backbones
from .static_model import StaticModel

RELATED_FLAGS_SETTING = {
    'FLAGS_cudnn_exhaustive_search': 1,
    'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
    'FLAGS_max_inplace_grad_add': 8,
}
paddle.fluid.set_flags(RELATED_FLAGS_SETTING)

def train(args):

    world_size = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    rank = int(os.getenv("PADDLE_TRAINER_ID", 0))

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = paddle.CUDAPlace(gpu_id)

    if world_size > 1:
        import paddle.distributed.fleet as fleet
        strategy = fleet.DistributedStrategy()
        strategy.without_graph_optimization = True
        fleet.init(is_collective=True, strategy=strategy)

    os.makedirs(args.output, exist_ok=True)
    init_logging(rank, args.output)
    writer = LogWriter(logdir=args.logdir)
    
    if args.dataset == 'synthetic':
        trainset = SyntheticDataset(args.num_classes)
    else:
        trainset = CommonDataset(root_dir=args.data_dir, label_file=args.label_file, is_bin=args.is_bin)
        
    num_image = len(trainset)    
    steps_per_epoch = num_image // args.batch_size
    total_batch_size = args.batch_size * world_size
    if args.train_unit == 'epoch':
        warmup_steps = steps_per_epoch * args.warmup_num
        total_steps = steps_per_epoch * args.train_num
        decay_steps = [x * steps_per_epoch for x in args.decay_boundaries]
        total_epoch = args.train_num
    else:
        warmup_steps = args.warmup_num
        total_steps = args.train_num
        decay_steps = [x for x in args.decay_boundaries]
        total_epoch = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    
    if rank == 0:
        print('total_batch_size: {}'.format(total_batch_size))
        print('warmup_steps: {}'.format(warmup_steps))
        print('steps_per_epoch: {}'.format(steps_per_epoch))
        print('total_steps: {}'.format(total_steps))
        print('total_epoch: {}'.format(total_epoch))
        
    base_lr = total_batch_size * args.lr / 512
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        paddle.optimizer.lr.PiecewiseDecay(
            boundaries=decay_steps,
            values=[base_lr * (args.lr_decay**i) for i in range(len(decay_steps) + 1)]),
        warmup_steps,
        0,
        base_lr)

    train_program = paddle.static.Program()
    test_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    
    margin_loss_params = eval("losses.{}".format(args.loss))()
    train_model = StaticModel(
        main_program=train_program,
        startup_program=startup_program,
        backbone_class_name=args.backbone,
        embedding_size=args.embedding_size,
        classifier_class_name=args.classifier,
        num_classes=args.num_classes,
        sample_ratio=args.sample_ratio,
        lr_scheduler=lr_scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        mode='train',
        fp16=args.fp16,
        fp16_configs={
            'init_loss_scaling': args.init_loss_scaling,
            'incr_every_n_steps': args.incr_every_n_steps,
            'decr_every_n_nan_or_inf': args.decr_every_n_nan_or_inf,
            'incr_ratio': args.incr_ratio,
            'decr_ratio': args.decr_ratio,
            'use_dynamic_loss_scaling': args.use_dynamic_loss_scaling,
            'use_pure_fp16': args.use_pure_fp16            
        },
        margin_loss_params=margin_loss_params,
    )
        
    if args.do_validation_while_train:
        test_model = StaticModel(
            main_program=test_program,
            startup_program=startup_program,
            backbone_class_name=args.backbone,
            embedding_size=args.embedding_size,
            mode='test',
        )
                    
        callback_verification = CallBackVerification(
            args.validation_interval_step,
            rank,
            world_size,
            args.batch_size,
            test_program,
            list(test_model.backbone.input_dict.values()),
            list(test_model.backbone.output_dict.values()),
            args.val_targets,
            args.data_dir
        )    
    
    callback_logging = CallBackLogging(
        args.log_interval_step,
        rank,
        world_size,
        total_steps,
        args.batch_size,
        writer
    )
    checkpoint = Checkpoint(
        rank=rank,
        world_size=world_size,
        embedding_size=args.embedding_size,
        num_classes=args.num_classes,
        model_save_dir=os.path.join(args.output, args.backbone),
        checkpoint_dir=args.checkpoint_dir,
        max_num_last_checkpoint=args.max_num_last_checkpoint
    )   
    
    exe = paddle.static.Executor(place)
    exe.run(startup_program)   
    
    start_epoch = 0
    global_step = 0
    loss_avg = AverageMeter()
    if args.resume:
        ckp = checkpoint.load(program=train_program, for_train=True)
        train_program.set_state_dict(ckp['state_dict'])
        start_epoch = ckp['extra_info']['epoch'] + 1
        lr_state = checkpoint['extra_info']['lr_state']
        # there last_epoch means last_step in for PiecewiseDecay
        # since we always use step style for lr_scheduler
        global_step = lr_state['last_epoch']
        lr_scheduler.set_state_dict(lr_state)
                
    train_loader = paddle.io.DataLoader(
        trainset,
        feed_list=list(train_model.backbone.input_dict.values()),
        places=place,
        return_list=False,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0) 
    
    for epoch in range(start_epoch, total_epoch):
        for step, data in enumerate(train_loader):
            global_step += 1
            loss_v = exe.run(train_program,
                feed=data,
                fetch_list=[train_model.classifier.output_dict['loss']],
                use_program_cache=True)
        
            loss_avg.update(np.array(loss_v)[0], 1)
            lr_value = train_model.optimizer.get_lr()
            callback_logging(global_step, loss_avg, epoch, lr_value)
            if args.validation_interval_step:
                callback_verification(global_step)
            train_model.lr_scheduler.step()
            sys.stdout.flush()
            if global_step >= total_steps:
                break

        checkpoint.save(train_program, lr_scheduler=train_model.lr_scheduler, epoch=epoch, for_train=True)
    writer.close()