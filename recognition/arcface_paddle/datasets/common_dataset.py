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

import pickle
import paddle
import os
import cv2
import six
import random
import paddle
import numpy as np
import logging
from PIL import Image
from io import BytesIO

from datasets.kv_helper import read_img_from_bin


def transform(img):
    # random horizontal flip
    if random.randint(0, 1) == 0:
        img = cv2.flip(img, 1)
    # normalize to mean 0.5, std 0.5
    img = (img - 127.5) * 0.00784313725
    # BGR2RGB
    img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    return img

class CommonDataset(paddle.io.Dataset):
    def __init__(self, root_dir, label_file, is_bin=True):
        super(CommonDataset, self).__init__()
        self.root_dir = root_dir
        self.label_file = label_file
        self.partial_lines = self.get_file_list(label_file)
        self.delimiter = "\t"
        self.is_bin = is_bin

        self.num_samples = len(self.partial_lines)

    def get_file_list(self, label_file):
        with open(label_file, "r") as fin:
            full_lines = fin.readlines()

        random.shuffle(full_lines)
        partial_lines = []
        rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        world_size = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        per_rank_lines = (len(full_lines) + world_size - 1) // world_size
        lack_lines = per_rank_lines * world_size - len(full_lines)
        full_lines += full_lines[0:lack_lines]
        start_line = rank * per_rank_lines
        end_line = (rank + 1) * per_rank_lines
        partial_lines = full_lines[start_line:end_line]

        logging.info("rank: {}/{}, finish reading file, image num: {}, from: {} to: {}, total num: {}"
            .format(rank, world_size, len(partial_lines), start_line, end_line, len(full_lines)))
        return partial_lines

    def __getitem__(self, idx):

        line = self.partial_lines[idx]

        img_path, label = line.strip().split(self.delimiter)
        img_path = os.path.join(self.root_dir, img_path)
        if self.is_bin:
            img = read_img_from_bin(img_path)
        else:
            img = cv2.imread(img_path)

        img = transform(img)
        
        img = img.astype('float32')
        label = np.int32(label)
        if paddle.fluid.framework.in_dygraph_mode():
            img = paddle.to_tensor(img, dtype='float32')
            label = paddle.to_tensor(label, dtype='int32')

        return img, label

    def __len__(self):
        return self.num_samples

class SyntheticDataset(paddle.io.Dataset):
    def __init__(self, num_classes):
        super(SyntheticDataset, self).__init__()
        self.num_classes = num_classes
        self.label_list = np.random.randint(0, num_classes, (100000,), dtype=np.int32)
        self.num_samples = len(self.label_list)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        img = transform(img)
        
        img = img.astype('float32')
        label = np.int32(label)
        if paddle.fluid.framework.in_dygraph_mode():
            img = paddle.to_tensor(img, dtype='float32')
            label = paddle.to_tensor(label, dtype='int32')

        return img, label

    def __len__(self):
        return self.num_samples
    
# 返回为 numpy
def load_bin(path, image_size):
    if six.PY2:
        bins, issame_list = pickle.load(open(path, 'rb'))
    else:
        bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0, 1]:
        data = np.empty(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        if six.PY2:
            if not isinstance(_bin, six.string_types):
                _bin = _bin.tostring()
            img_ori = Image.open(StringIO(_bin))
        else:
            img_ori = Image.open(BytesIO(_bin))
        for flip in [0, 1]:
            img = img_ori.copy()
            if flip == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img).astype('float32').transpose((2, 0, 1))
            img = (img - 127.5) * 0.00784313725
            data_list[flip][i][:] = img
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list