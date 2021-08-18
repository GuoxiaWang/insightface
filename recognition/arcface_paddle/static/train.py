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
import logging

import paddle
from visualdl import LogWriter

from utils.logging import AverageMeter, CallBackLogging
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

def gather_optimization_pass(program, weight_name):
    def check_contains(name, name_list):
        for n in name_list:
            if name in n:
                return True
        return False

    op_idxs = []
    gather_grad_op = None
    momentum_op = None
    for idx, op in enumerate(program.global_block().ops):
        if (op.type == 'gather_grad' or op.type == 'momentum') and check_contains(weight_name, op.input_arg_names):
            op_idxs.append(idx)
            if op.type == 'momentum':
                momentum_op = op
            if op.type == 'gather_grad':
                gather_grad_op = op  
       
    if gather_grad_op is not None and momentum_op is not None:
        for idx, op in enumerate(program.global_block().ops):
            if (op.type == 'update_loss_scaling' or op.type == 'check_finite_and_unscale'):
                input_idxs = []
                input_arg_names = op.input("X")
                for i, name in enumerate(input_arg_names):
                    if '@GRAD' in name and weight_name in name:
                        input_idxs.append(i)
                if len(input_idxs) > 0:
                    for i in reversed(input_idxs):
                        input_arg_names.pop(i)
                    op.desc.set_input("X", input_arg_names)

                output_idxs = []
                output_arg_names = op.output("Out")
                for i, name in enumerate(output_arg_names):
                    if '@GRAD' in name and weight_name in name:
                        output_idxs.append(i)
                if len(output_idxs) > 0:
                    for i in reversed(output_idxs):
                        output_arg_names.pop(i)
                    op.desc.set_output("Out", output_arg_names)

                if op.type == 'check_finite_and_unscale':
                    op_role_idxs = []
                    op_role_var = op.attr("op_role_var")
                    for i, name in enumerate(op_role_var):
                        if '@GRAD' in name and weight_name in name:
                            op_role_idxs.append(i)
                    if len(op_role_idxs) > 0:
                        for i in reversed(op_role_idxs):
                            op_role_var.pop(i)
                        op.desc._set_attr("op_role_var", op_role_var)
                    
        inputs = {
           'Param': momentum_op.input('Param'),
           'Velocity': momentum_op.input('Velocity'),
           'LearningRate': momentum_op.input('LearningRate'),
           'Grad': gather_grad_op.input('Out@GRAD'),
           'Index': gather_grad_op.input('Index'),
           'Axis': gather_grad_op.input('Axis'),
        }
        outputs = {
           'ParamOut': momentum_op.output('ParamOut'),
           'VelocityOut': momentum_op.output('VelocityOut'),
        }
        if 'MasterParam' in momentum_op.input_names and len(momentum_op.input('MasterParam')) > 0:
            inputs['MasterParam'] = momentum_op.input('MasterParam')
        if 'MasterParamOut' in momentum_op.output_names and len(momentum_op.output('MasterParamOut')) > 0:
            outputs['MasterParamOut'] =  momentum_op.output('MasterParamOut')
        
        attrs = {
           'mu': momentum_op.attr('mu'),
           'use_nesterov': momentum_op.attr('use_nesterov'),
           'regularization_method': momentum_op.attr('regularization_method'),
           'regularization_coeff': momentum_op.attr('regularization_coeff'),
           'multi_precision': momentum_op.attr('multi_precision'),
           'rescale_grad': momentum_op.attr('rescale_grad'),
           'op_device': momentum_op.attr('op_device'),
           'op_namescope': momentum_op.attr('op_namescope'),
           'op_role': momentum_op.attr('op_role'),
           'op_role_var': momentum_op.input('Param'),
           'axis': gather_grad_op.attr('axis'),
        }
        program.global_block()._insert_op(
           op_idxs[-1] + 1,
           type='sparse_momentum',
           inputs=inputs,
           outputs=outputs,
           attrs=attrs)
        
        for idx in reversed(op_idxs):
            program.global_block()._remove_op(idx, sync=False)

        var_names = []
        for idx, name in enumerate(program.global_block().vars):
            if '@GRAD' in name and weight_name in name:
                var_names.append(name)
        for name in var_names:
            program.global_block()._remove_var(name, sync=False)
        program.global_block()._sync_with_cpp()

def train(args):

    writer = LogWriter(logdir=args.logdir)
    
    rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
    world_size = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    
    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = paddle.CUDAPlace(gpu_id)

    if world_size > 1:
        import paddle.distributed.fleet as fleet
        strategy = fleet.DistributedStrategy()
        strategy.without_graph_optimization = True
        fleet.init(is_collective=True, strategy=strategy)

    if args.use_synthetic_dataset:
        trainset = SyntheticDataset(args.num_classes)
    else:
        trainset = CommonDataset(root_dir=args.data_dir, label_file=args.label_file, is_bin=args.is_bin)
        
    num_image = len(trainset)    
    total_batch_size = args.batch_size * world_size
    steps_per_epoch = num_image // total_batch_size
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
        logging.info('world_size: {}'.format(world_size))
        logging.info('total_batch_size: {}'.format(total_batch_size))
        logging.info('warmup_steps: {}'.format(warmup_steps))
        logging.info('steps_per_epoch: {}'.format(steps_per_epoch))
        logging.info('total_steps: {}'.format(total_steps))
        logging.info('total_epoch: {}'.format(total_epoch))
        logging.info('decay_steps: {}'.format(decay_steps))
        
    base_lr = total_batch_size * args.lr / 512
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=decay_steps,
        values=[base_lr * (args.lr_decay**i) for i in range(len(decay_steps) + 1)])
    if warmup_steps > 0:
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            lr_scheduler,
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
    
    if args.sample_ratio < 1.0:
        gather_optimization_pass(train_program, 'dist@fc@rank')
    
    if rank == 0:
        with open(os.path.join(args.output, 'main_program.txt'), 'w') as f:
            f.write(str(train_program))
        
    if rank == 0 and args.do_validation_while_train:
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
        extra_info = checkpoint.load(program=train_program, for_train=True)
        start_epoch = extra_info['epoch'] + 1
        lr_state = extra_info['lr_state']
        # there last_epoch means last_step in for PiecewiseDecay
        # since we always use step style for lr_scheduler
        global_step = lr_state['last_epoch']
        train_model.lr_scheduler.set_state_dict(lr_state)
                
    train_loader = paddle.io.DataLoader(
        trainset,
        feed_list=list(train_model.backbone.input_dict.values()),
        places=place,
        return_list=False,
        num_workers=args.num_workers,
        batch_sampler=paddle.io.DistributedBatchSampler(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )
    )
    
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
            if rank == 0 and args.do_validation_while_train:
                callback_verification(global_step)
            train_model.lr_scheduler.step()
            
            if global_step >= total_steps:
                break
            sys.stdout.flush()

        checkpoint.save(train_program, lr_scheduler=train_model.lr_scheduler, epoch=epoch, for_train=True)
    writer.close()
