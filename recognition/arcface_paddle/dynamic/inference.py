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
import numpy as np
import cv2
import paddle

from .utils.io import Checkpoint
from . import backbones

def transform(img):
    # normalize to mean 0.5, std 0.5
    img = (img - 127.5) * 0.00784313725
    # BGR2RGB
    img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    return img.astype('float32')

def inference(args):
    checkpoint = Checkpoint(
        rank=0,
        world_size=1,
        embedding_size=args.embedding_size,
        num_classes=None,
        checkpoint_dir=args.checkpoint_dir,
    ) 
    
    backbone = eval("backbones.{}".format(args.backbone))(num_features=args.embedding_size)
    checkpoint.load(backbone, for_train=False)
    backbone.eval()
    
    if os.path.exists(args.img_path):
        img = cv2.imread(img)
    else:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    img = cv2.resize(img, (112, 112))
    img = transform(img)
    img = np.expand_dims(img, 0)
    img = paddle.to_tensor(img)
    feat = backbone(img).numpy()
    
    print(feat)

