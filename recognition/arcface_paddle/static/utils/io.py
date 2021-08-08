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
import logging
import numpy as np

class Checkpoint(object):
    def __init__(self,
                 rank,
                 world_size,
                 embedding_size,
                 num_classes,
                 model_save_dir="./",
                 checkpoint_dir=None,
                 max_num_last_checkpoint=3):
        
        self.rank: int = rank
        self.world_size: int = world_size
        self.embedding_size: int = embedding_size
        self.num_classes: int = num_classes
        self.model_save_dir: str = model_save_dir
        self.checkpoint_dir: str = checkpoint_dir
        self.max_num_last_checkpoint: int = max_num_last_checkpoint
         
    def save(self, program, lr_scheduler=None, epoch=0, for_train=True):
        model_save_dir = os.path.join(self.model_save_dir, str(epoch))
        if not os.path.exists(model_save_dir):
            # may be more than one processes trying
            # to create the directory
            try:
                os.makedirs(model_save_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

        param_state_dict = program.state_dict(mode='param')
        for name, param in param_state_dict.items():
            # for non dist param, we only save their at trainer 0,
            # but for dist param, we need to save their at all trainers
            if 'dist@' in name and '@rank@' in name or self.rank == 0:
                paddle.save(param,
                            os.path.join(model_save_dir, name + '.pdparam'))

        if for_train:
            opt_state_dict = program.state_dict(mode='opt')
            for name, opt in opt_state_dict.items():
                # for non opt var, we only save their at trainer 0,
                # but for opt var, we need to save their at all trainers
                if 'dist@' in name and '@rank@' in name or self.rank == 0:
                    paddle.save(opt,
                                os.path.join(model_save_dir, name + '.pdopt'))

            if self.rank == 0:
                # save some extra info for resume
                # pretrain_nranks, emb_dim, num_classes are used for
                # re-split fc weight when gpu setting changed.
                # epoch use to restart.
                config_file = os.path.join(model_save_dir, 'meta.json')
                extra_info = dict()
                extra_info["pretrain_world_size"] = self.world_size
                extra_info["embedding_size"] = self.embedding_size
                extra_info['num_classes'] = self.num_classes
                extra_info['epoch'] = epoch
                extra_info['lr_state'] = lr_scheduler.state_dict()
                with open(config_file, 'w') as f:
                    json.dump(extra_info, f)

        logger.info("Save model to {}.".format(self.model_save_dir))
        if self.rank == 0 and self.max_num_last_checkpoint > 0:
            for idx in range(-1, epoch - self.max_num_last_checkpoint + 1):
                path = os.path.join(self.model_save_dir, str(idx))
                if os.path.exists(path):
                    logger.info("Remove checkpoint {}.".format(path))
                    shutil.rmtree(path)
                    
    def load(self, program, for_train=True):
        assert os.path.exists(self.checkpoint_dir)
        checkpoint_dir = os.path.abspath(self.checkpoint_dir)

        logger.info("Load checkpoint from '{}'. ".format(checkpoint_dir))

        state_dict = {}
        dist_weight_state_dict = {}
        dist_weight_velocity_state_dict = {}
        dist_bias_state_dict = {}
        dist_bias_velocity_state_dict = {}
        for path in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, path)
            if not os.path.isfile(path):
                continue

            basename = os.path.basename(path)
            name, ext = os.path.splitext(basename)

            if ext not in ['.pdopt', '.pdparam']:
                continue

            if not for_train and ext == '.pdopt':
                continue

            tensor = paddle.load(path, return_numpy=True)

            if 'dist@' in name and '@rank@' in name:
                if '.w' in name and 'velocity' not in name:
                    dist_weight_state_dict[name] = tensor
                elif '.w' in name and 'velocity' in name:
                    dist_weight_velocity_state_dict[name] = tensor
                elif '.b' in name and 'velocity' not in name:
                    dist_bias_state_dict[name] = tensor
                elif '.b' in name and 'velocity' in name:
                    dist_bias_velocity_state_dict[name] = tensor

            else:
                state_dict[name] = tensor

        if for_train:
            meta_file = os.path.join(checkpoint_dir, 'meta.json')
            if not os.path.exists(meta_file):
                logger.error(
                    "Please make sure the checkpoint dir {} exists, and "
                    "parameters in that dir are validating.".format(
                        checkpoint_dir))
                exit()

            with open(meta_file, 'r') as handle:
                config = json.load(handle)
                
        # Preporcess distributed parameters.
        if self.world_size > 1:
            pretrain_world_size = config['pretrain_world_size']
            assert pretrain_world_size > 0
            embedding_size = config['embedding_size']
            assert embedding_size == self.embedding_size
            num_classes = config['num_classes']
            assert num_classes == self.num_classes

            logger.info("Parameters for pre-training: pretrain_world_size ({}), "
                        "embedding_size ({}), and num_classes ({}).".format(
                            pretrain_nranks, emb_dim, num_classes))
            logger.info("Parameters for inference or fine-tuning: "
                        "world_size ({}).".format(self.world_size))

            rank_str = '%05d' % self.rank

            dist_weight_state_dict = self.rearrange_weight(
                dist_weight_state_dict, pretrain_world_size, self.world_size)
            dist_bias_state_dict = self.rearrange_weight(
                dist_bias_state_dict, pretrain_world_size, self.world_size)
            for name, value in dist_weight_state_dict.items():
                if rank_str in name:
                    state_dict[name] = value
            for name, value in dist_bias_state_dict.items():
                if rank_str in name:
                    state_dict[name] = value

            if for_train:
                dist_weight_velocity_state_dict = self.rearrange_weight(
                    dist_weight_velocity_state_dict, pretrain_world_size,
                    self.world_size)
                dist_bias_velocity_state_dict = self.rearrange_weight(
                    dist_bias_velocity_state_dict, pretrain_world_size,
                    self.world_size)
                for name, value in dist_weight_velocity_state_dict.items():
                    if rank_str in name:
                        state_dict[name] = value
                for name, value in dist_bias_velocity_state_dict.items():
                    if rank_str in name:
                        state_dict[name] = value
        if for_train:
            return {'state_dict': state_dict, 'extra_info': config}
        else:
            return {'state_dict': state_dict}
        
    def rearrange_weight(self, weight_dict, init_num_rank, new_num_rank):
        """
        A help function to convert pre-trained distributed fc parameters for
        inference or fine-tuning. Note that the number of ranks or GPUs for
        inference or fine-tuning can be different from that for pre-training.

        Args:
            weight_dict(dict): the dict store distributed parameters,
                key: eg. dist@fc@rank@00000.w_0
                value: numpy.ndarray
            init_num_rank(int) : pre-trained weight at init_num_rank gpu device.
            new_num_rank(int) : want to rearrange weight to new_num_rank gpu device.

        Returns:
            dict: rearranged weight for new_num_rank gpu device.
        """

        ret_dict = {}
        if init_num_rank == new_num_rank:
            return weight_dict

        if len(weight_dict) == 0:
            return weight_dict

        # generate name format
        name_format = list(weight_dict.keys())[0]
        name_format = name_format.split('.')
        name_format[0] = name_format[0].split('@')
        name_format[0][-1] = '%05d'
        name_format[0] = '@'.join(name_format[0])
        name_format = '.'.join(name_format)

        # calculate num class of pretrain shard
        # num class of new shard
        num_class = sum([
            w.shape[1] if len(w.shape) == 2 else len(w)
            for _, w in weight_dict.items()
        ])
        init_nshard = (num_class + init_num_rank - 1) // init_num_rank
        new_nshard = (num_class + new_num_rank - 1) // new_num_rank

        if new_nshard * (new_num_rank - 1) >= num_class:
            raise ValueError(
                "num class {} cann't be rationally splited by num rank {}".format(
                    num_class, new_num_rank))

        if init_num_rank > new_num_rank:
            for new_idx in range(new_num_rank):
                start = new_idx * new_nshard
                end = min((new_idx + 1) * new_nshard - 1, num_class - 1)
                init_shard_idx_start = start // init_nshard
                init_shard_idx_end = end // init_nshard

                weight_list = []
                for init_idx in range(init_shard_idx_start,
                                      init_shard_idx_end + 1):
                    name = name_format % init_idx
                    init_weight = weight_dict[name]
                    s = max(start - init_idx * init_nshard, 0)
                    if init_idx == init_shard_idx_end:
                        e = min(end - init_idx * init_nshard + 1, init_nshard)
                    else:
                        e = init_nshard
                    if len(init_weight.shape) == 2:
                        weight_list.append(init_weight[:, s:e])
                    else:
                        weight_list.append(init_weight[s:e])

                name = name_format % new_idx
                # for 2-dimention, we concat at axis=1,
                # else for 1-dimention, we concat at axis=0
                ret_dict[name] = np.concatenate(
                    weight_list, axis=len(weight_list[0].shape) - 1)
        else:
            for new_idx in range(new_num_rank):
                start = new_idx * new_nshard
                end = min((new_idx + 1) * new_nshard - 1, num_class - 1)
                init_shard_idx_start = start // init_nshard
                init_shard_idx_end = end // init_nshard

                if init_shard_idx_start == init_shard_idx_end:
                    name = name_format % init_shard_idx_start
                    init_weight = weight_dict[name]
                    init_start = init_shard_idx_start * init_nshard
                    s = max(start - init_start, 0)
                    e = min((init_shard_idx_start + 1) * init_nshard,
                            end) - init_start + 1
                    if len(init_weight.shape) == 2:
                        new_weight = init_weight[:, s:e]
                    else:
                        new_weight = init_weight[s:e]
                else:
                    # init_shard_idx_start + 1 == init_shard_idx_end
                    name = name_format % init_shard_idx_start
                    init_weight = weight_dict[name]
                    init_start = init_shard_idx_start * init_nshard
                    s = max(start - init_start, 0)
                    if len(init_weight.shape) == 2:
                        new_weight = init_weight[:, s:]
                    else:
                        new_weight = init_weight[s:]

                    e = end - (init_shard_idx_end * init_nshard) + 1
                    if e > 0:
                        name = name_format % init_shard_idx_end
                        init_weight = weight_dict[name]
                        if len(init_weight.shape) == 2:
                            new_weight2 = init_weight[:, :e]
                        else:
                            new_weight2 = init_weight[:e]

                        new_weight = np.concatenate(
                            [new_weight, new_weight2],
                            axis=len(new_weight.shape) - 1)
                name = name_format % new_idx
                ret_dict[name] = new_weight

        return ret_dict