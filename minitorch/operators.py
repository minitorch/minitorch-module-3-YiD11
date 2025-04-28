"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def id(a: float) -> float:
    return a

def neg(a: float) -> float:
    """Negate a number."""
    return -a

def inv(a: float) -> float:
    """Inverse of a number."""
    if a == 0:
        raise ZeroDivisionError("Division by zero is not allowed.")
    return 1.0 / a

def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    
    if a == 0:
        raise ZeroDivisionError("Division by zero is not allowed.")
    return -b / (a * a)

def lt(a: float, b: float) -> bool:
    """Less than."""
    return a < b

def eq(a: float, b: float) -> bool:
    """Equal."""
    return a == b

def max(a: float, b: float) -> float:
    """Maximum of two numbers."""
    return a if a > b else b

def is_close(a: float, b: float) -> bool:
    """Check if two numbers are close."""
    return abs(a - b) < 1e-2

def sigmoid(a: float) -> float:
    """Sigmoid function."""
    return 1 / (math.exp(-a) + 1)

def relu(a: float) -> float:
    """ReLU function."""
    return a if a > 0 else 0.0

def log(a: float) -> float:
    """Natural logarithm."""
    return math.log(a)

def exp(a: float) -> float:
    """Exponential function."""
    return math.exp(a)

def log_back(a: float, b: float) -> float:
    """Backpropagation for logarithm."""
    return b / a

def relu_back(a: float, b: float) -> float:
    """Backpropagation for ReLU."""
    return b if a > 0 else 0.0

# ## Task 0.3
# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(f: Callable[[float], float], lst: Iterable[float]) -> list:
    """Map a function over a list."""
    return [f(x) for x in lst]

def zipWith(f: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]) -> list:
    """Zip two lists with a function."""
    return [f(x, y) for x, y in zip(lst1, lst2)]

def reduce(f: Callable[[float, float], float], lst: Iterable[float]) -> float:
    """Reduce a list with a function."""
    result = 0
    for i, x in enumerate(lst):
        if i == 0:
            result = x
        else:
            result = f(result, x)
    return result

def negList(lst: Iterable[float]) -> list:
    """Negate a list."""
    return map(neg, lst)

def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> list:
    """Add two lists together."""
    return zipWith(add, lst1, lst2)

def sum(lst: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, lst)

def prod(lst: Iterable[float]) -> float:
    """Take the product of a list."""
    return reduce(mul, lst)
