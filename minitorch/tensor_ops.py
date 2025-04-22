from __future__ import annotations

from functools import cache
import itertools
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

import numpy as np
import numpy.typing as npt 
from typing_extensions import Protocol, Sequence

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
    import itertools
    return np.sum(
        np.array(
            list(itertools.product(*[np.arange(i) for i in shape])),
            dtype=np.int32,
        ) * strides,
        axis=-1,
    )

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
    in_shape_extend = np.concatenate([[1] * diff_num, in_shape], axis=0) if diff_num > 0 else in_shape
    in_strides_extend = np.concatenate([[in_strides[0]] * diff_num, in_strides], axis=0) if diff_num > 0 else in_strides

    diff_indices = np.where(in_shape_extend != out_shape)[0]
    same_indices = np.where(in_shape_extend == out_shape)[0]
    split_index = len(diff_indices)
    assert np.all(in_shape_extend[diff_indices] == 1), f"input shape: {in_shape_extend} cannot broadcast to output shape: {out_shape}"
    
    in_shape_extend = np.concatenate([in_shape_extend[diff_indices], in_shape_extend[same_indices]], axis=0)
    in_strides_extend = np.concatenate([in_strides_extend[diff_indices], in_strides_extend[same_indices]], axis=0)
    
    out_shape = np.concatenate([out_shape[diff_indices], out_shape[same_indices]], axis=0)
    out_strides = np.concatenate([out_strides[diff_indices], out_strides[same_indices]], axis=0)
    
    in_right_indices = index_permutation(in_shape_extend[split_index:], in_strides_extend[split_index:])
    out_right_indices = index_permutation(out_shape[split_index:], out_strides[split_index:])

    if split_index > 0:
        in_left_indices = index_permutation(in_shape_extend[:split_index], in_strides_extend[:split_index])
        out_left_indices = index_permutation(out_shape[:split_index], out_strides[:split_index])
        in_indices = np.sum(
            np.array(list(
                itertools.product(in_left_indices, in_right_indices),
            )),
            axis=-1,
        )
        out_indices = np.sum(
            np.array(list(
                itertools.product(out_left_indices, out_right_indices),
            )),
            axis=-1,
        )
    else:
        in_indices = in_right_indices
        out_indices = out_right_indices
    k = len(out_indices) // len(in_indices)
    if k > 1:
        in_indices = np.tile(in_indices, k)

    return out_indices, in_indices

def tensor_map(fn: Callable[[float], float]) -> Any:
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
        # assert len(np.unique(in_indices)) == len(in_indices) and len(np.unique(out_indices)) == len(out_indices), f"in_shape = {in_shape}, in_strides = {in_strides}, in_indices = {in_indices}, out_shape = {out_shape}, out_strides = {out_strides}, out_indices = {out_indices}, "
        out[out_indices] = np.vectorize(fn)(in_storage[in_indices])
    
    # def _map_(
    #     out: Storage,
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     in_storage: Storage,
    #     in_shape: Shape,
    #     in_strides: Strides,
    # ) -> None:
    #     if np.equal(out_shape, in_shape).all():
    #         _simple_map(0, len(out_shape), [], out, out_shape, out_strides, in_storage, in_shape, in_strides)
    #     elif len(in_shape) <= len(out_shape) and shape_broadcast(in_shape, out_shape) == out_shape:
    #         _broadcasted_map(0, len(out_shape), out, [], out_shape, out_strides, in_storage, [], in_shape, in_strides)
            
    # def _simple_map(
    #     dep: int,
    #     max_dep: int,
    #     index: Sequence[int],
    #     out: Storage,
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     in_storage: Storage,
    #     in_shape: Shape,
    #     in_strides: Strides,
    # ):
    #     if dep == max_dep:
    #         out_pos = index_to_position(index, out_strides)
    #         in_pos = index_to_position(index, in_strides)
    #         out[out_pos] = fn(in_storage[in_pos])
    #         return

    #     for i in range(out_shape[dep]):
    #         index.append(i)
    #         _simple_map(dep + 1, max_dep, index, out, out_shape, out_strides, in_storage, in_shape, in_strides)
    #         index.pop()

    # def _broadcasted_map(
    #     dep: int,
    #     max_dep: int,
    #     out_storage: Storage,
    #     out_index: Sequence[int],
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     in_storage: Storage,
    #     in_index: Sequence[int],
    #     in_shape: Shape,
    #     in_strides: Strides,
    # ):
    #     diff = _shape_size_diff(out_shape, in_shape)
    #     if dep == max_dep:
    #         out_pos = index_to_position(out_index, out_strides)
    #         in_pos = index_to_position(in_index, in_strides)
    #         out_shape[out_pos] = fn(in_storage[in_pos])
    #         return
        
    #     for i in range(out_shape[dep]):
    #         if dep < diff:
    #             out_index.append(i)
    #             _broadcasted_map(dep + 1, max_dep, out_storage, out_index, out_shape, out_strides, in_storage, in_index, in_shape, in_strides)
    #             out_index.pop()
    #         elif in_shape[dep] == 1:
    #             out_index.append(i)
    #             in_index.append(0)
    #             _broadcasted_map(dep + 1, max_dep, out_storage, out_index, out_shape, out_strides, in_storage, in_index, in_shape, in_strides)
    #             out_index.pop()
    #             in_index.pop()
    #         else:
    #             assert in_shape[dep] == out_shape[dep], f"Shape mismatch: input shape: {in_shape}, output shape: {out_shape}"
    #             out_index.append(i)
    #             in_index.append(i)
    #             _broadcasted_map(dep + 1, max_dep, out_storage, out_index, out_shape, out_strides, in_storage, in_index, in_shape, in_strides)
    #             out_index.pop()
    #             in_index.pop()
    
    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
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

    
    # def _zip_(
    #     out: Storage,
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     a_storage: Storage,
    #     a_shape: Shape,
    #     a_strides: Strides,
    #     b_storage: Storage,
    #     b_shape: Shape,
    #     b_strides: Strides,
    # ) -> None:
    #     if np.equal(out_shape, a_shape).all() and np.equal(out_shape, b_shape).all():
    #         _simple_zip(0, len(out_shape), [], out, out_shape, out_strides, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides)
    #     elif len(a_shape) <= len(out_shape) and shape_broadcast(a_shape, out_shape) == out_shape:
    #         assert len(a_shape) == len(b_shape), f"Shape mismatch: input shape: {a_shape}, b shape: {b_shape}"
    #         _broadcasted_zip(0, len(out_shape), out, [], out_shape, out_strides, a_storage, [], a_shape, a_strides, b_storage, [], b_shape, b_strides)

    # def _simple_zip(
    #     dep: int,
    #     max_dep: int,
    #     index: List[int],
    #     out_storage: Storage,
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     a_storage: Storage,
    #     a_shape: Shape,
    #     a_strides: Strides,
    #     b_storage: Storage,
    #     b_shape: Shape,
    #     b_strides: Strides,
    # ):
    #     if dep == max_dep:
    #         out_pos = index_to_position(index, out_strides)
    #         a_pos = index_to_position(index, a_strides)
    #         b_pos = index_to_position(index, b_strides)
    #         out_storage[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
    #         return
        
    #     for i in range(out_shape[dep]):
    #         index.append(i)
    #         _simple_zip(dep + 1, max_dep, index, out_storage, out_shape, out_strides, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides)
    #         index.pop()
    
    # def _broadcasted_zip(
    #     dep: int,
    #     max_dep: int,
    #     out_storage: Storage,
    #     out_index: Sequence[int],
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     a_storage: Storage,
    #     a_index: Sequence[int],
    #     a_shape: Shape,
    #     a_strides: Strides,
    #     b_storage: Storage,
    #     b_index: Sequence[int],
    #     b_shape: Shape,
    #     b_strides: Strides,
    # ):
    #     diff = _shape_size_diff(out_shape, a_shape)
    #     if dep == max_dep:
    #         out_pos = index_to_position(out_index, out_strides)
    #         a_pos = index_to_position(a_index, a_strides)
    #         b_pos = index_to_position(b_index, b_strides)
    #         out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
    #         return
        
    #     for i in range(out_shape[dep]):
    #         if dep < diff:
    #             out_index.append(i)
    #             _broadcasted_zip(dep + 1, max_dep, out_storage, out_index, out_shape, out_strides, a_storage, a_index, a_shape, a_strides, b_storage, b_index, b_shape, b_strides)
    #             out_index.pop()
    #         elif a_shape[dep] == 1:
    #             out_index.append(i)
    #             a_index.append(0)
    #             b_index.append(0)
    #             _broadcasted_zip(dep + 1, max_dep, out_storage, out_index, out_shape, out_strides, a_storage, a_index, a_shape, a_strides, b_storage, b_index, b_shape, b_strides)
    #             out_index.pop()
    #             a_index.pop()
    #             b_index.pop()
    #         else:
    #             assert a_shape[dep] == out_shape[dep], f"Shape mismatch: input shape: {a_shape}, output shape: {out_shape}"
    #             out_index.append(i)
    #             a_index.append(i)
    #             b_index.append(i)
    #             _broadcasted_zip(dep + 1, max_dep, out_storage, out_index, out_shape, out_strides, a_storage, a_index, a_shape, a_strides, b_storage, b_index, b_shape, b_strides)
    #             out_index.pop()
    #             a_index.pop()
    #             b_index.pop()

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
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
        
        
    # def _reduce_(
    #     out: Storage,
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     a_storage: Storage,
    #     a_shape: Shape,
    #     a_strides: Strides,
    #     reduce_dim: int,
    # ) -> None:
    #     _recursive_reduce(
    #         0,
    #         len(out_shape),
    #         out,
    #         [],
    #         out_shape,
    #         out_strides,
    #         a_storage,
    #         [],
    #         a_shape,
    #         a_strides,
    #         reduce_dim,
    #     )

    # def _recursive_reduce(
    #     dep: int,
    #     max_dep: int,
    #     out_storage: Storage,
    #     out_index: List[int],
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     in_storage: Storage,
    #     in_index: List[int],
    #     in_shape: Shape,
    #     in_strides: Strides,
    #     reduce_dim: int,
    # ):
    #     if dep == max_dep:
    #         out_pos = index_to_position(out_index, out_strides)
    #         for i in range(in_shape[reduce_dim]):
    #             in_index[reduce_dim] = i
    #             in_pos = index_to_position(in_index, in_strides)
    #             if i == 0:
    #                 out_storage[out_pos] = in_storage[in_pos]
    #             else:
    #                 out_storage[out_pos] = fn(out_storage[out_pos], in_storage[in_pos])
    #         return

    #     for i in range(out_shape[dep]):
    #         in_index.append(i)
    #         out_index.append(i)
    #         _recursive_reduce(dep + 1, max_dep, out_storage, out_index, out_shape, out_strides, in_storage, in_index, in_shape, in_strides, reduce_dim)
    #         out_index.pop()
    #         in_index.pop()

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
