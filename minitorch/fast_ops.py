from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import (
    MapProto,
    TensorOps,
    index_broadcast,
    index_permutation,
    fast_index_broadcast,
)

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)

class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        fn = njit(fn)
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        fn = njit(fn)
        f = tensor_zip(fn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        fn = njit(fn)
        f = tensor_reduce(fn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if (len(out_shape) == len(in_shape)
            and len(out_strides) == len(in_strides)
            and np.equal(out_shape, in_shape).all()
            and np.equal(out_strides, in_strides).all()
        ):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
            return
        
        out_indices, in_indices = fast_index_broadcast(out_shape, out_strides, in_shape, in_strides)
        for i in prange(len(out_indices)):
            out[out_indices[i]] = fn(in_storage[in_indices[i]]) # type: ignore

    # return _map
    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """
    
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if (len(out_shape) == len(a_shape)
            and len(out_shape) == len(b_shape)
            and len(out_strides) == len(a_strides)
            and len(out_strides) == len(b_strides)
            and np.equal(out_shape, a_shape).all()
            and np.equal(out_shape, b_shape).all()
            and np.equal(out_strides, a_strides).all()
            and np.equal(out_strides, b_strides).all()
        ):
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
            return
        
        out_indices, a_indices = fast_index_broadcast(out_shape, out_strides, a_shape, a_strides)
        out_indices2, b_indices = fast_index_broadcast(out_shape, out_strides, b_shape, b_strides)
        order = np.argsort(out_indices)
        out_indices = out_indices[order]
        a_indices = a_indices[order]
        b_indices = b_indices[np.argsort(out_indices2)]
        for i in prange(len(out_indices)):
            out[out_indices[i]] = fn(a_storage[a_indices[i]], b_storage[b_indices[i]]) # type: ignore

    # return _zip
    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        if len(a_shape) == 1:
            for i in range(a_shape[0]):
                out[0] = fn(out[0], a_storage[i])
            return
            
        a_broad_shape = np.zeros(len(a_shape) - 1, dtype=np.int32)
        a_broad_strides = np.ones(len(a_strides) - 1, dtype=np.int32)
        a_broad_shape[:reduce_dim] = a_shape[:reduce_dim]
        a_broad_strides[:reduce_dim] = a_strides[:reduce_dim]
        a_broad_shape[reduce_dim:] = a_shape[reduce_dim + 1:]
        a_broad_strides[reduce_dim:] = a_strides[reduce_dim + 1:]
        
        a_indices = index_permutation(a_broad_shape, a_broad_strides)
        out_indices = index_permutation(out_shape, out_strides)
        for i in range(a_shape[reduce_dim]):
            a_reduce_indices = a_indices + np.ones_like(a_indices) * i * a_strides[reduce_dim]
            for j in range(len(out_indices)):
                out[out_indices[j]] = fn(out[out_indices[j]], a_storage[a_reduce_indices[j]]) # type: ignore

    # return _reduce
    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    out_outer_indices, a_outer_indices = fast_index_broadcast(out_shape[:-2], out_strides[:-2], a_shape[:-2], a_strides[:-2])
    out_outer_indices2, b_outer_indices = fast_index_broadcast(out_shape[:-2], out_strides[:-2], b_shape[:-2], b_strides[:-2])
    order = np.argsort(out_outer_indices)
    out_outer_indices = out_outer_indices[order]
    a_indices = a_outer_indices[order]
    b_indices = b_outer_indices[np.argsort(out_outer_indices2)]

    for i in prange(len(out_outer_indices)):
        out_outer_index = out_outer_indices[i]
        a_outer_index = a_outer_indices[i]
        b_outer_index = b_outer_indices[i]
        for j in prange(out_shape[-2]):
            for k in prange(out_shape[-1]):
                out_inner_index = j * out_strides[-2] + k * out_strides[-1]
                a_indices = np.arange(a_shape[-1], dtype=np.int32) * a_strides[-1] + j * a_strides[-2] + a_outer_index
                b_indices = np.arange(b_shape[-2], dtype=np.int32) * b_strides[-2] + k * b_strides[-1] + b_outer_index
                out[out_outer_index + out_inner_index] = np.sum(
                    a_storage[a_indices] * b_storage[b_indices]
                )

# tensor_matrix_multiply = _tensor_matrix_multiply
tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
