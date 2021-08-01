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

from dataloader import CommonDataset

from paddle.io import DataLoader
from config import config as cfg
from classifier import LargeScaleClassifier
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter
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

def main(args):


    world_size = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    rank = int(os.getenv("PADDLE_TRAINER_ID", 0))

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = paddle.CUDAPlace(gpu_id)

    if world_size > 1:
        strategy = fleet.DistributedStrategy()
        strategy.without_graph_optimization = True
        fleet.init(is_collective=True, strategy=strategy)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        time.sleep(2)

    writer = LogWriter(logdir=args.logdir)
    trainset = CommonDataset(root_dir=cfg.data_dir, label_file=cfg.file_list, is_bin=args.is_bin)
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        places=place,
        shuffle=True,
        drop_last=True,
        num_workers=0)

    backbone = eval("backbones.{}".format(args.network))()
    backbone.train()

    clip_by_norm = ClipGradByNorm(5.0)
    margin_loss_params = eval("losses.{}".format(args.loss))()

    large_scale_classifier = LargeScaleClassifier(
        rank=0,
        world_size=1,
        resume=0,
        batch_size=args.batch_size,
        margin1=margin_loss_params.margin1,
        margin2=margin_loss_params.margin2,
        margin3=margin_loss_params.margin3,
        scale=margin_loss_params.scale,
        num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate,
        embedding_size=args.embedding_size,
        prefix=args.output)

    lr_scheduler_decay = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.lr, lr_lambda=cfg.lr_func, verbose=True)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=lr_scheduler_decay,
        warmup_steps=cfg.warmup_epoch,
        start_lr=0,
        end_lr=args.lr / 512 * args.batch_size,
        verbose=True)
    optimizer = paddle.optimizer.Momentum(
        parameters=[{
            'params': backbone.parameters(),
        }, {
            'params': large_scale_classifier.parameters(),
        }],
        learning_rate=lr_scheduler,
        momentum=0.9,
        weight_decay=args.weight_decay,
        grad_clip=clip_by_norm)

    if world_size > 1:
        optimizer = fleet.distributed_optimizer(optimizer)

    start_epoch = 0
    total_step = int(
        len(trainset) / args.batch_size / world_size * cfg.num_epoch)
    if rank == 0:
        print("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(2000, rank, cfg.val_targets,
                                                 cfg.data_dir)
    callback_logging = CallBackLogging(10, rank, total_step, args.batch_size,
                                       world_size, writer)
    callback_checkpoint = CallBackModelCheckpoint(rank, args.output,
                                                  args.network)

    loss = AverageMeter()
    global_step = 0
    for epoch in range(start_epoch, cfg.num_epoch):
        for step, (img, label) in enumerate(train_loader):
            label = label.flatten()
            global_step += 1
            sys.stdout.flush()
            features = backbone(img)
            loss_v = large_scale_classifier(features, label, optimizer)
            loss_v.backward()
            optimizer.step()
            optimizer.clear_grad()

            large_scale_classifier.update()

            lr_value = optimizer._global_learning_rate().numpy()[0]
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, lr_value)
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, large_scale_classifier)
        lr_scheduler.step()
    writer.close()


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    
    parser = argparse.ArgumentParser(description='Paddle ArcFace Training')
    parser.add_argument(
        '--network',
        type=str,
        default='MobileFaceNet_128',
        help='backbone network')
    parser.add_argument(
        '--loss', type=str, default='ArcFace', help='loss function')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument(
        '--batch_size', type=int, default=512, help='batch size')
    parser.add_argument(
        '--weight_decay', type=float, default=2e-4, help='weight decay')
    parser.add_argument(
        '--embedding_size', type=int, default=128, help='embedding size')
    parser.add_argument('--logdir', type=str, default='./log', help='log dir')
    parser.add_argument(
        '--output', type=str, default='emore_arcface', help='output dir')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    parser.add_argument('--is_bin', type=str2bool, default=True, help='whether the train data is bin or original image file')
    args = parser.parse_args()
    main(args)
