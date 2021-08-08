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
config.backbone = 'FresResNet100'
config.classifier = 'LargeScaleClassifier'
config.embedding_size = 512
config.model_parallel = True
config.sample_ratio = 0.1
config.loss = 'ArcFace'

config.fp16 = True
config.init_loss_scaling = 1.0
config.incr_every_n_steps = 2000
config.decr_every_n_nan_or_inf = 2
config.incr_ratio = 2.0
config.decr_ratio = 0.5
config.use_dynamic_loss_scaling = True
config.use_pure_fp16 = False

config.lr = 0.1 # for global batch size = 512
config.lr_decay = 0.1
config.weight_decay = 2e-4
config.momentum = 0.9
config.train_unit = 'step' # 'step' or 'epoch'
config.warmup_num = 1000
config.train_num = 180000
config.decay_boundaries = [100000, 140000, 160000]

config.dataset = "emore"
config.data_dir = "./MS1M_bin"
config.label_file = "./MS1M_binlabel.txt"
config.is_bin = True
config.num_classes = 85742
config.batch_size = 64 # 512 for 1 GPU
config.num_workers = 4

config.do_validation_while_train = True
config.validation_interval_step = 2000
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.logdir = './log'
config.log_interval_step = 10
config.output = './emore_arcface'
config.resume = False
config.checkpoint_dir = None
config.max_num_last_checkpoint = 3
