# Copyright 2022 MetaOPT Team. All Rights Reserved.
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
# ==============================================================================

from torchopt._src.alias import adam
from torchopt._src.optimizer.meta.base import MetaOptimizer
from torchopt._src.typing import ScalarOrSchedule


class MetaAdam(MetaOptimizer):
    """The classic Adam optimizer."""

    def __init__(
        self,
        net,
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        moment_requires_grad: bool = True,
        use_accelerated_op: bool = False
    ):
        """The `init` function.

        Args:
            net (nn.Module):
                A network whose parameters should be optimized.
            args:
                Other arguments see `alias.adam`, here we set `moment_requires_grad=True`
                to make tensors like momentum be differentiable.
        """

        super().__init__(
            net,
            adam(
                lr=lr,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                moment_requires_grad=moment_requires_grad,
                use_accelerated_op=use_accelerated_op
            )
        )