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
# This file is modified from:
# https://github.com/google/jaxopt/blob/main/jaxopt/_src/linear_solve.py
# ==============================================================================
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Linear algebra solver for ``A x = b`` using conjugate gradient."""

# pylint: disable=invalid-name

import functools
from typing import Callable, Optional

from torchopt import linalg
from torchopt.linear_solve.utils import make_ridge_matvec
from torchopt.typing import TensorTree


__all__ = ['solve_cg']


def _solve_cg(
    matvec: Callable[[TensorTree], TensorTree],  # (x) -> A @ x
    b: TensorTree,
    ridge: Optional[float] = None,
    init: Optional[TensorTree] = None,
    **kwargs,
) -> TensorTree:
    """Solves ``A x = b`` using conjugate gradient.

    It assumes that ``A`` is a hermitian, positive definite matrix.

    Args:
        matvec: A function that returns the product between ``A`` and a vector.
        b: A tree of tensors for the right hand side of the equation.
        ridge: Optional ridge regularization.
        init: Optional initialization to be used by conjugate gradient.
        **kwargs: Additional keyword arguments for the conjugate gradient solver.

    Returns:
        The solution with the same structure as ``b``.
    """
    if ridge is not None:
        #      (x) -> A @ x + ridge * x
        # i.e. (x) -> (A + ridge * I) @ x
        matvec = make_ridge_matvec(matvec, ridge=ridge)

    # Returns solution for `(A + ridge * I) @ x = b`.
    return linalg.cg(matvec, b, x0=init, **kwargs)


def solve_cg(**kwargs):
    """Wrapper for :func:`solve_cg`."""
    return functools.partial(_solve_cg, **kwargs)