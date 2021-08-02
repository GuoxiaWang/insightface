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

import os
import paddle
import paddle.nn as nn
from paddle.nn.functional import normalize, linear
import pickle


class LargeScaleClassifier(nn.Layer):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @paddle.no_grad()
    def __init__(self,
                 rank,
                 world_size,
                 batch_size,
                 resume,
                 num_classes,
                 margin1=1.0,
                 margin2=0.5,
                 margin3=0.0,
                 scale=64.0,
                 sample_rate=1.0,
                 embedding_size=512,
                 prefix="./"):
        super(LargeScaleClassifier, self).__init__()
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.sample_rate: float = sample_rate
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        self.num_local: int = (num_classes + world_size - 1) // world_size
        if num_classes % world_size != 0 and rank == world_size - 1:
            self.num_local = num_classes % self.num_local
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.logit_scale = scale

        self.weight_name = os.path.join(
            self.prefix, "rank:{}_softmax_weight.pkl".format(self.rank))
        self.weight_mom_name = os.path.join(
            self.prefix, "rank:{}_softmax_weight_mom.pkl".format(self.rank))

        if resume:
            try:
                self.weight: paddle.Tensor = paddle.load(self.weight_name)
                print("softmax weight resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight = paddle.normal(0, 0.01, (self.num_local,
                                                      self.embedding_size))
                print("softmax weight resume fail!")

            try:
                self.weight_mom: paddle.Tensor = paddle.load(
                    self.weight_mom_name)
                print("softmax weight mom resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight_mom: paddle.Tensor = paddle.zeros_like(self.weight)
                print("softmax weight mom resume fail!")
        else:
            self.weight = paddle.normal(0, 0.01,
                                        (self.num_local, self.embedding_size))
            self.weight_mom: paddle.Tensor = paddle.zeros_like(self.weight)
            print("softmax weight init successfully!")
            print("softmax weight mom init successfully!")

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = paddle.create_parameter(
                shape=self.weight.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(self.weight))
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = paddle.create_parameter(
                shape=[1, 1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.empty((1, 1))))

    def save_params(self):
        with open(self.weight_name, 'wb') as file:
            pickle.dump(self.weight.numpy(), file)
        with open(self.weight_mom_name, 'wb') as file:
            pickle.dump(self.weight_mom.numpy(), file)

    @paddle.no_grad()
    def update(self):
        self.weight[self.index] = self.sub_weight
        self.weight_mom[self.index] = self.sub_weight_mom

    @paddle.no_grad()
    def sample(self, total_label, optimizer):
        if int(self.sample_rate) != 1:
            # partial fc sample process
            total_label, self.index = paddle.class_center_sample(
                total_label, self.num_local, self.num_sample)
            # TODO(GuoxiaWang)
            sub_weight_tensor = self.weight[self.index]
            self.sub_weight = paddle.create_parameter(
                shape=sub_weight_tensor.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(sub_weight_tensor))
            self.sub_weight_mom = self.weight_mom[self.index]
            optimizer._accumulators['velocity'].pop(optimizer._parameter_list[-1]['params'][0], None)
            optimizer._parameter_list[-1]['params'][0] = self.sub_weight
            optimizer._accumulators['velocity'][self.sub_weight.name] = self.sub_weight_mom
            # END TODO
        return total_label

    def forward(self, feature, label, optimizer):
        if self.world_size > 1:
            feature_list = []
            paddle.distributed.all_gather(feature_list, feature)
            total_feature = paddle.concat(feature_list, axis=0)

            label_list = []
            paddle.distributed.all_gather(label_list, label)
            total_label = paddle.concat(label_list, axis=0)
        else:
            total_feature = feature
            total_label = label

        total_label.stop_gradient = True
        total_label = self.sample(total_label, optimizer)

        norm_feature = normalize(total_feature)
        norm_weight = normalize(self.sub_weight)
        logits = linear(norm_feature, paddle.t(norm_weight))

        loss_v = paddle.nn.functional.margin_softmax_with_cross_entropy(
            logits,
            total_label,
            margin1=self.margin1,
            margin2=self.margin2,
            margin3=self.margin3,
            scale=self.logit_scale,
            return_softmax=False)
        return loss_v


