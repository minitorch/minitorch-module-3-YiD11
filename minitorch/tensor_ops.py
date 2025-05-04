from __future__ import annotations

from functools import cache
import itertools
from turtle import shapesize
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type, TypeVar

import numpy as np
import numpy.typing as npt 
from typing_extensions import Protocol, Sequence

from numba import prange
from numba import njit as _njit

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

Fn = TypeVar("Fn")
def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore

class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]: ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.

def shape_size_diff(out_shape: Shape, in_shape: Shape) -> int:
    return abs(len(out_shape) - len(in_shape))

def index_permutation(shape: Shape, strides: Strides) -> npt.NDArray[np.int32]:
    """
    Generates a permutation of indices for a given shape and computes their corresponding 
    linear indices based on the provided strides.

    Args:
        shape (Sequence): A sequence of integers representing the dimensions of the tensor.
        strides (Sequence): A sequence of integers representing the strides for each dimension.

    Returns:
        Sequence[Sequence[int]]: A sequence of linear indices corresponding to the permutations 
        of the input shape, calculated using the provided strides.

    Example:
        Given a shape of (2, 3) and strides of (3, 1), this function will generate all 
        permutations of indices for the shape (e.g., (0, 0), (0, 1), ..., (1, 2)) and compute 
        their linear indices based on the strides.
    """
    permut_size = np.prod(shape)
    dim_size = len(shape)
    shape_cumprod = np.cumprod(shape)
    indices = np.arange(permut_size)
    ret = np.zeros((permut_size,), dtype=np.int32)
    for i in range(dim_size):
        real_stride = shape_cumprod[-1] // shape_cumprod[i] # strides may not match the shape
        ret[indices] += (indices // real_stride) % shape[i] * strides[i]
    return ret

def index_permutation_pair(index1: npt.NDArray[np.int32], index2: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """
    Return:
        Permutation of input
    """
    n1, n2 = index1.shape[0], index2.shape[0]
    n = n1 * n2
    ret = np.zeros((n), dtype=np.int32)
    ret_index_permut = np.arange(n)
    ret[ret_index_permut] = index1[ret_index_permut // n2 % n1] + index2[ret_index_permut % n2]
    return ret


def index_broadcast(
        out_shape: Shape,
        out_strides: Strides,
        in_shape: Shape,
        in_strides: Strides,
) -> tuple[Index, Index]:
    """
    Computes the broadcasted indices for input and output tensors based on their shapes and strides.
        This function calculates the indices required to map elements from an input tensor
        to an output tensor when broadcasting is applied. It ensures that the input tensor
        can be broadcast to match the output tensor's shape and computes the corresponding
        indices for both tensors.
    Args:
        out_shape (Shape): The shape of the output tensor.
        out_strides (Strides): The strides of the output tensor.
        in_shape (Shape): The shape of the input tensor.
        in_strides (Strides): The strides of the input tensor.
    Returns:
        tuple[Sequence[int], Sequence[int]]: A tuple containing two sequences:
            - The indices for the output tensor.
            - The indices for the input tensor.
    Raises:
        AssertionError: If the input shape cannot be broadcast to the output shape.
    Notes:
        - Broadcasting rules are applied to align the input tensor's shape with the
            output tensor's shape.
        - The function handles cases where the input tensor's shape has fewer dimensions
            than the output tensor's shape by extending the input shape with ones.
        - The indices are computed in a way that respects the broadcasting semantics.
    Example:
        Given an input tensor of shape (1, 3) and an output tensor of shape (2, 3),
        this function computes the indices required to broadcast the input tensor
        to match the output tensor.
    """
    diff_num = shape_size_diff(out_shape, in_shape)
    in_shape_extend = np.ones((diff_num + len(in_shape)), dtype=np.int32)
    in_shape_extend[diff_num:] = in_shape
    in_strides_extend = np.zeros((diff_num + len(in_shape)), dtype=np.int32)
    in_strides_extend[diff_num:] = in_strides    

    diff_indices = np.where(in_shape_extend != out_shape)[0]
    same_indices = np.where(in_shape_extend == out_shape)[0]
    
    diff_indices_size = len(diff_indices)
    same_indices_size = len(same_indices)
    
    in_shape_reorder    = np.empty_like(in_shape_extend, dtype=np.int32)
    in_strides_reorder  = np.empty_like(in_strides_extend, dtype=np.int32)
    out_shape_reorder   = np.empty_like(out_shape, dtype=np.int32)
    out_strides_reorder = np.empty_like(out_strides, dtype=np.int32)
    
    in_shape_reorder    [np.arange(diff_indices_size)]                      = in_shape_extend   [diff_indices]
    in_shape_reorder    [np.arange(same_indices_size) + diff_indices_size]  = in_shape_extend   [same_indices]
    in_strides_reorder  [np.arange(diff_indices_size)]                      = in_strides_extend [diff_indices]
    in_strides_reorder  [np.arange(same_indices_size) + diff_indices_size]  = in_strides_extend [same_indices]
    out_shape_reorder   [np.arange(diff_indices_size)]                      = out_shape         [diff_indices]
    out_shape_reorder   [np.arange(same_indices_size) + diff_indices_size]  = out_shape         [same_indices]
    out_strides_reorder [np.arange(diff_indices_size)]                      = out_strides       [diff_indices]
    out_strides_reorder [np.arange(same_indices_size) + diff_indices_size]  = out_strides       [same_indices]

    if same_indices_size == 0:
        in_left_indices = index_permutation(in_shape_reorder[:diff_indices_size], in_strides_reorder[:diff_indices_size])
        out_left_indices = index_permutation(out_shape_reorder[:diff_indices_size], out_strides_reorder[:diff_indices_size])
        in_indices = in_left_indices
        out_indices = out_left_indices
    elif diff_indices_size == 0:
        in_right_indices = index_permutation(in_shape_reorder[diff_indices_size:], in_strides_reorder[diff_indices_size:])
        out_right_indices = index_permutation(out_shape_reorder[diff_indices_size:], out_strides_reorder[diff_indices_size:])
        in_indices = in_right_indices
        out_indices = out_right_indices
    else:
        in_left_indices = index_permutation(in_shape_reorder[:diff_indices_size], in_strides_reorder[:diff_indices_size])
        in_right_indices = index_permutation(in_shape_reorder[diff_indices_size:], in_strides_reorder[diff_indices_size:])
        
        out_left_indices = index_permutation(out_shape_reorder[:diff_indices_size], out_strides_reorder[:diff_indices_size])
        out_right_indices = index_permutation(out_shape_reorder[diff_indices_size:], out_strides_reorder[diff_indices_size:])

        in_indices = index_permutation_pair(in_left_indices, in_right_indices)
        out_indices = index_permutation_pair(out_left_indices, out_right_indices)

    in_aligned_indices = np.zeros((out_indices.shape[0],), dtype=np.int32)
    for i in range(out_indices.shape[0] // in_indices.shape[0]):
        in_aligned_indices[i * len(in_indices) : (i + 1) * len(in_indices)] = in_indices
    return out_indices, in_aligned_indices

def tensor_map(
    fn: Callable[[float], float],
    ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:

    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_indices, in_indices = index_broadcast(out_shape, out_strides, in_shape, in_strides)
        out[out_indices] = np.vectorize(fn)(in_storage[in_indices])
    
    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
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
        out_indices, a_indices = index_broadcast(out_shape, out_strides, a_shape, a_strides)
        _, b_indices = index_broadcast(out_shape, out_strides, b_shape, b_strides)
        out[out_indices] = np.vectorize(fn)(a_storage[a_indices], b_storage[b_indices])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
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
        assert len(a_shape) == len(out_shape) and out_shape[reduce_dim] == 1, f"Shape mismatch: input shape: {a_shape}, output shape: {out_shape}"
        a_indices = index_permutation(
            np.concatenate([a_shape[:reduce_dim], a_shape[reduce_dim + 1:]], axis=0),
            np.concatenate([a_strides[:reduce_dim], a_strides[reduce_dim + 1:]], axis=0),
        )
        out_indices = index_permutation(out_shape, out_strides)
        for i in range(a_shape[reduce_dim]):
            a_reduce_indices = a_indices + np.ones_like(a_indices) * i * a_strides[reduce_dim]
            out[out_indices] = np.vectorize(fn)(out[out_indices], a_storage[a_reduce_indices])
        return

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)