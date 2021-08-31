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

from paddle.amp import GradScaler
from paddle import _C_ops

class LSCGradScaler(GradScaler):
    def __init__(self,
             enable=True,
             init_loss_scaling=2.**15,
             incr_ratio=2.0,
             decr_ratio=0.5,
             incr_every_n_steps=1000,
             decr_every_n_nan_or_inf=2,
             use_dynamic_loss_scaling=True):
        super(LSCGradScaler, self).__init__(enable, init_loss_scaling, incr_ratio,
                                     decr_ratio, incr_every_n_steps,
                                     decr_every_n_nan_or_inf,
                                     use_dynamic_loss_scaling)

    def step(self, optimizer, classifier=None):
        if not self._enable:
            if classifier is not None:
                classifier.step(optimizer)
            return optimizer.step()

        #  unscale the grad
        self._unscale(optimizer)
        
        if not self._found_inf and classifier is not None and len(classifier._parameter_list) > 0:
            param_grads = [
                param._grad_ivar() for param in classifier._parameter_list
                if param._grad_ivar() is not None
            ]

            _C_ops.check_finite_and_unscale(
                param_grads,
                self._scale,
                param_grads,
                self._found_inf)

        if self._found_inf:
            self._cache_founf_inf = True
        else:
            optimizer.step()
            if classifier is not None:
                classifier.step(optimizer)
                
            self._cache_founf_inf = False

        if self._use_dynamic_loss_scaling:
            # update the scale
            self._update()