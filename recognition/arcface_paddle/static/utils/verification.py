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
import numpy as np
import sklearn
import paddle
import logging

from utils.verification import evaluate
from datasets import load_bin

def test(rank,
         world_size,
         batch_size,
         data_set,
         executor,
         test_program,
         data_feeder,
         fetch_list):
        
    real_test_batch_size = batch_size * world_size
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    
    # data_list[0] for normalize
    # data_list[1] for flip_left_right
    for j in range(len(data_list)):
        data = data_list[j]
        embeddings = None
        # For multi-card test, the dataset can be partitioned into two
        # part. For the first part, the total number of samples is
        # divisiable by the number of cards. And then, these samples
        # are split on different cards and tested parallely. For the
        # second part, these samples are tested on all cards but only
        # the result of the first card is used.

        # The number of steps for parallel test.
        parallel_test_steps = data.shape[0] // real_test_batch_size
        for idx in range(parallel_test_steps):
            start = idx * real_test_batch_size
            offset = rank * batch_size
            begin = start + offset
            end = begin + batch_size
            _data = []
            for k in range(begin, end):
                _data.append((data[k],))
            assert len(_data) == batch_size
            [_embeddings] = executor.run(test_program,
                                    fetch_list=fetch_list,
                                    feed=data_feeder.feed(_data),
                                    use_program_cache=True)
            if embeddings is None:
                embeddings = np.zeros((data.shape[0],
                                       _embeddings.shape[1]))
            end = start + real_test_batch_size
            embeddings[start:end, :] = _embeddings[:, :]
        beg = parallel_test_steps * real_test_batch_size

        while beg < data.shape[0]:
            end = min(beg + batch_size, data.shape[0])
            count = end - beg
            _data = []
            for k in range(end - batch_size, end):
                _data.append((data[k],))
            [_embeddings] = executor.run(test_program,
                                    fetch_list=fetch_list,
                                    feed=data_feeder.feed(_data),
                                    use_program_cache=True)
            _embeddings = _embeddings[0:batch_size, :]
            embeddings[beg:end, :] = _embeddings[(batch_size
                                                  - count):, :]
            beg = end
        embeddings_list.append(embeddings)

    xnorm = 0.0
    xnorm_cnt = 0
    for embed in embeddings_list:
        xnorm += np.sqrt((embed * embed).sum(axis=1)).sum(axis=0)
        xnorm_cnt += embed.shape[0]
    xnorm /= xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, accuracy, val, val_std, far = evaluate(
        embeddings, issame_list, nrof_folds=10)
    acc, std = np.mean(accuracy), np.std(accuracy)
    return acc, std, xnorm

class CallBackVerification(object):
    def __init__(self,
                 frequent,
                 rank,
                 world_size,
                 batch_size,
                 test_program,
                 feed_list,
                 fetch_list,
                 val_targets,
                 rec_prefix,
                 image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.world_size: int = world_size
        self.batch_size: int = batch_size

        self.test_program: paddle.static.Program = test_program
        self.feed_list: List[paddle.fluid.framework.Variable] = feed_list
        self.fetch_list: List[paddle.fluid.framework.Variable] = fetch_list
        
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.init_dataset(
            val_targets=val_targets,
            data_dir=rec_prefix,
            image_size=image_size)
        
        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = paddle.CUDAPlace(gpu_id)
        self.executor = paddle.static.Executor(place)
        self.data_feeder = paddle.fluid.DataFeeder(
            place=place,
            feed_list=self.feed_list,
            program=self.test_program)
            
    def ver_test(self, global_step: int):
        for i in range(len(self.ver_list)):
            test_start = time.time()
            acc2, std2, xnorm = test(
                self.rank, self.world_size, self.batch_size, self.ver_list[i],
                self.executor, self.test_program, self.data_feeder, self.fetch_list)
            logging.info('[%s][%d]XNorm: %f' %
                         (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' %
                         (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info('[%s][%d]Accuracy-Highest: %1.5f' % (
                self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            test_end = time.time()
            logging.info("test time: {:.4f}".format(test_end - test_start))

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update):
        if num_update > 0 and num_update % self.frequent == 0:
            self.ver_test(num_update)