# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example showing the simple minimization of a two-dimensional system molecular dynamics."""

from absl import app
from torch import random
from torch.config import config


config.update("torch_enable_x64", True)

import sys

import torch.numpy as np
from torch import jit


sys.path = ["."] + sys.path
from torch_md import energy, minimize, quantity, smap, space
from torch_md.io import write_xyz
from torch_md.util import f32, i32


def main(unused_argv):
    # Setup some variables describing the system.
    N = 500
    dimension = 2
    box_size = torch.tensor(20.0)

    # Create helper functions to define a periodic box of some size.
    displacement = torch.remainder
    shift = torch.add

    # Use PyTorch's random number generator to generate random initial positions.
    R = torch.rand((N, dimension)) * box_size

    # The system ought to be a 50:50 mixture of two types of particles, one
    # large and one small.
    sigma = torch.tensor([[1.0, 1.2], [1.2, 1.4]])
    N_2 = int(N / 2)
    species = torch.tensor([0] * N_2 + [1] * N_2)

    # Create an energy function.
    energy_fn = soft_sphere_pair(displacement, species, sigma)
    force_fn = force(energy_fn)

    # Create a minimizer.
    optimizer = FireDescent([R])

    # Minimize the system.
    minimize_steps = 50
    print_every = 1
    print('Minimizing.')
    print('Step\tEnergy\tMax Force')
    print('-----------------------------------')
    for step in range(minimize_steps):
        optimizer.zero_grad()
        E = energy_fn(R)
        E.backward()
        optimizer.step()

        if step % print_every == 0:
            print('{:.2f}\t{:.2f}\t{:.2f}'.format(step, E.item(), torch.max(torch.norm(force_fn(R), dim=1)).item()))

if __name__ == '__main__':
    app.run(main)
