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
from paddle.nn.functional import normalize
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
                 sample_ratio=1.0,
                 embedding_size=512,
                 prefix="./"):
        super(LargeScaleClassifier, self).__init__()
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.sample_ratio: float = sample_ratio
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        self.num_local: int = (num_classes + world_size - 1) // world_size
        if num_classes % world_size != 0 and rank == world_size - 1:
            self.num_local = num_classes % self.num_local
        self.num_sample: int = int(self.sample_ratio * self.num_local)
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.logit_scale = scale

        self.weight_name = os.path.join(
            self.prefix, "rank:{}_softmax_weight.pkl".format(self.rank))
        self.weight_mom_name = os.path.join(
            self.prefix, "rank:{}_softmax_weight_mom.pkl".format(self.rank))
        
        self.weight = self.create_parameter(
            shape=(self.num_local, self.embedding_size),
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal(std=0.01))

    def sample(self, total_label, optimizer):
        # partial fc sample process
        with paddle.fluid.dygraph.no_grad():
            total_label, self.index = paddle.class_center_sample(
                total_label, self.num_local, self.num_sample)

        self.sub_weight = self.weight[self.index]
        self.sub_weight.stop_gradient = False

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

        if int(self.sample_ratio) != 1:
            total_label = self.sample(total_label, optimizer)
        total_label.stop_gradient = True

        norm_feature = normalize(total_feature)
        norm_weight = normalize(self.sub_weight)
        logits = paddle.matmul(norm_feature, norm_weight, transpose_y=True)

        loss = paddle.nn.functional.margin_softmax_with_cross_entropy(
            logits,
            total_label,
            margin1=self.margin1,
            margin2=self.margin2,
            margin3=self.margin3,
            scale=self.logit_scale,
            return_softmax=False)
        
        return loss
