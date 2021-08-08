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
from ..utils import utils_argparse as parser
import paddle
import backbones
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
        print('Load checkpoint from {}.'.format(checkpoint_path))
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
        sample_ratio=args.sample_ratio,
        embedding_size=args.embedding_size,
        prefix=args.output)
    large_scale_classifier.train()

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

    optimizer = paddle.optimizer.Momentum(
        parameters=[{
            'params': backbone.parameters(),
        }, {
            'params': large_scale_classifier.parameters(),
        }],
#        parameters=backbone.parameters(),
        learning_rate=lr_scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        grad_clip=clip_by_norm)

    print("large_scale_classifier.parameters(): ", large_scale_classifier.parameters())
    if world_size > 1:
        optimizer = fleet.distributed_optimizer(optimizer)

    callback_verification = CallBackVerification(args.do_validation_while_train, args.validation_interval_step,
                                                 rank, args.val_targets, args.data_dir)
    callback_logging = CallBackLogging(args.log_interval_step, rank, total_steps, args.batch_size,
                                       world_size, writer)
    callback_checkpoint = CallBackModelCheckpoint(rank, args.output,
                                                  args.network)

    loss = AverageMeter()
    global_step = 0
    start_epoch = 0
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
#            large_scale_classifier.update(optimizer.get_lr())
            optimizer.clear_grad()

            lr_value = optimizer._global_learning_rate().numpy()[0]
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, lr_value)
            callback_verification(global_step, backbone)
            lr_scheduler.step()

        callback_checkpoint(global_step, backbone, large_scale_classifier)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
