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

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse

def str2bool(v):
    return str(v).lower() in ("true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser(description='Paddle Face Inference')

    # Model setting
    parser.add_argument(
        '--is_static', type=str2bool, default='False', help='whether to use static mode')    
    parser.add_argument(
        '--backbone', type=str, default='FresResNet50', help='backbone network')
    parser.add_argument(
        '--embedding_size', type=int, default=512, help='embedding size')
    parser.add_argument(
        '--checkpoint_dir', type=str, default='MS1M_v3_arcface/FresResNet50/24/', help='checkpoint direcotry')
    parser.add_argument(
        '--img_path', type=str, default='', help='test image path')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.is_static:
        from static.inference import inference
        paddle.enable_static()
    else:
        from dynamic.inference import inference
        
    inference(args)   
