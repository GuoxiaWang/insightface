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

from easydict import EasyDict as edict

config = edict()
config.sample_rate = 0.1
config.momentum = 0.9

config.data_dir = "/wangguoxia/plsc/MS1M_v2/"
config.file_list = "/wangguoxia/plsc/MS1M_v2/label.txt"
config.num_classes = 85742
config.lr = 0.1 # for global batch size = 512
config.lr_decay = 0.1
config.train_unit = 'step' # 'step' or 'epoch'
config.warmup_num = 1000
config.train_num = 180000
config.decay_boundaries = [100000, 140000, 160000]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
