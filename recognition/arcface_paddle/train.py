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

from datasets import CommonDataset, SyntheticDataset

from paddle.io import DataLoader
from config import config as cfg
from classifier import LargeScaleClassifier
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from paddle.nn import ClipGradByNorm
from visualdl import LogWriter
import paddle
import backbones
import argparse
import losses
import time
import os
import sys

RELATED_FLAGS_SETTING = {
    'FLAGS_cudnn_exhaustive_search': 1,
    'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
    'FLAGS_max_inplace_grad_add': 8,
}
paddle.fluid.set_flags(RELATED_FLAGS_SETTING)


def print_args(args):
    print('\n--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('------------------------\n')

def main(args):


    world_size = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    rank = int(os.getenv("PADDLE_TRAINER_ID", 0))

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = paddle.CUDAPlace(gpu_id)

    if world_size > 1:
        import paddle.distributed.fleet as fleet
        from utils.utils_data_parallel import sync_gradients

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
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        places=place,
        shuffle=True,
        drop_last=True,
        num_workers=0)

    backbone = eval("backbones.{}".format(args.network))(num_features=args.embedding_size)
    if args.resume:
        checkpoint_path = os.path.join(args.output, args.network + '.pdparams')
        param_state_dict = paddle.load(checkpoint_path)
        backbone.set_dict(param_state_dict)
    backbone.train()

    clip_by_norm = ClipGradByNorm(5.0)
    margin_loss_params = eval("losses.{}".format(args.loss))()

    large_scale_classifier = LargeScaleClassifier(
        rank=rank,
        world_size=world_size,
        resume=args.resume,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        margin1=margin_loss_params.margin1,
        margin2=margin_loss_params.margin2,
        margin3=margin_loss_params.margin3,
        scale=margin_loss_params.scale,
        sample_rate=args.sample_rate,
        embedding_size=args.embedding_size,
        prefix=args.output)
    large_scale_classifier.train()

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

    base_lr = total_batch_size * args.lr / 512
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        paddle.optimizer.lr.PiecewiseDecay(
            boundaries=decay_steps,
            values=[base_lr * (args.lr_decay**i) for i in range(len(decay_steps) + 1)]),
        warmup_steps,
        0,
        base_lr)

    optimizer = paddle.optimizer.Momentum(
        parameters=[{
            'params': backbone.parameters(),
        }, {
            'params': large_scale_classifier.parameters(),
        }],
        learning_rate=lr_scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        grad_clip=clip_by_norm)

    if world_size > 1:
        optimizer = fleet.distributed_optimizer(optimizer)

    start_epoch = 0
    if rank == 0:
        print("Total Step is: %d" % total_steps)

    callback_verification = CallBackVerification(args.do_validation_while_train, args.validation_interval_step,
                                                 rank, args.val_targets, args.data_dir)
    callback_logging = CallBackLogging(args.log_interval_step, rank, total_steps, args.batch_size,
                                       world_size, writer)
    callback_checkpoint = CallBackModelCheckpoint(rank, args.output,
                                                  args.network)

    loss = AverageMeter()
    global_step = 0
    for epoch in range(start_epoch, total_epoch):
        for step, (img, label) in enumerate(train_loader):
            label = label.flatten()
            global_step += 1
            sys.stdout.flush()
            features = backbone(img)
            loss_v = large_scale_classifier(features, label, optimizer)
            loss_v.backward()
            if world_size > 1:
                # data parallel sync backbone gradients
                sync_gradients(backbone.parameters())
                
            optimizer.step()
            optimizer.clear_grad()

            large_scale_classifier.update()

            lr_value = optimizer._global_learning_rate().numpy()[0]
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, lr_value)
            callback_verification(global_step, backbone)
            lr_scheduler.step()

        callback_checkpoint(global_step, backbone, large_scale_classifier)
    writer.close()


if __name__ == '__main__':
    def str2bool(v):
        return str(v).lower() in ("true", "t", "1")

    def tostrlist(v):
        if isinstance(v, list):
            return v
        elif isinstance(v, str):
            return [e.strip() for e in v.split(',')]

    def tointlist(v):
        if isinstance(v, list):
            return v
        elif isinstance(v, str):
            return [int(e.strip()) for e in v.split(',')]
    
    parser = argparse.ArgumentParser(description='Paddle Face Training')

    # Model setting
    parser.add_argument(
        '--network', type=str, default=cfg.network, help='backbone network')
    parser.add_argument(
        '--embedding_size', type=int, default=cfg.embedding_size, help='embedding size')
    parser.add_argument(
        '--model_parallel', type=str2bool, default=cfg.model_parallel, help='whether to use model parallel')
    parser.add_argument(
        '--sample_rate', type=float, default=cfg.sample_rate, help='sample rate, use partial fc sample if sample rate less than 1.0')
    parser.add_argument(
        '--loss', type=str, default=cfg.loss, help='loss function')

    # Optimizer setting
    parser.add_argument(
        '--lr', type=float, default=cfg.lr, help='learning rate')
    parser.add_argument(
        '--lr_decay', type=float, default=cfg.lr_decay, help='learning rate decay factor')
    parser.add_argument(
        '--weight_decay', type=float, default=cfg.weight_decay, help='weight decay')
    parser.add_argument(
        '--momentum', type=float, default=cfg.momentum, help='sgd momentum')
    parser.add_argument(
        '--train_unit', type=str, default=cfg.train_unit, help='train unit, "step" or "epoch"')
    parser.add_argument(
        '--warmup_num', type=int, default=cfg.warmup_num, help='warmup num according train unit')
    parser.add_argument(
        '--train_num', type=int, default=cfg.train_num, help='train num according train unit')
    parser.add_argument(
        '--decay_boundaries', type=tointlist, default=cfg.decay_boundaries, help='piecewise decay boundaries')

    # Train dataset setting
    parser.add_argument(
        '--dataset', type=str, default=cfg.dataset, help='train dataset name')
    parser.add_argument(
        '--data_dir', type=str, default=cfg.data_dir, help='train dataset directory')
    parser.add_argument(
        '--label_file', type=str, default=cfg.label_file, help='train label file name, each line split by "\t"')
    parser.add_argument(
        '--is_bin', type=str2bool, default=cfg.is_bin, help='whether the train data is bin or original image file')
    parser.add_argument(
        '--num_classes', type=int, default=cfg.num_classes, help='classes of train dataset')
    parser.add_argument(
        '--batch_size', type=int, default=cfg.batch_size, help='batch size of each rank')

    # Validation dataset setting
    parser.add_argument(
        '--do_validation_while_train', type=str2bool, default=cfg.do_validation_while_train, help='do validation while train')
    parser.add_argument(
        '--validation_interval_step', type=int, default=cfg.validation_interval_step, help='validation interval step')
    parser.add_argument(
        '--val_targets', type=tostrlist, default=cfg.val_targets, help='val targets, list or str split by comma')

    # IO setting
    parser.add_argument(
        '--logdir', type=str, default=cfg.logdir, help='log dir')
    parser.add_argument(
        '--log_interval_step', type=int, default=cfg.log_interval_step, help='log interval step')
    parser.add_argument(
        '--output', type=str, default=cfg.output, help='output dir')
    parser.add_argument(
        '--resume', type=str2bool, default=cfg.resume, help='model resuming')

    args = parser.parse_args()
    print_args(args)
    main(args)
