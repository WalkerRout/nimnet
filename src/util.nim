
import neo
import math
import sequtils

proc sigmoid(x: float64): float64 =
  result = 1 / (1 + (-x).exp)

proc sigmoid_prime(x: float64): float64 =
  result = sigmoid(x) * (1 - sigmoid(x))

proc relu(x: float64): float64 =
  result = max(0.0, x)

proc relu_prime(x: float64): float64 =
  result = if x > 0.0: 1.0 else: 0.0

proc activation*(x: float64): float64 =
  result = relu(x)

proc activation_prime*(x: float64): float64 =
  result = relu_prime(x)

proc activation*(xs: Matrix[float64]): Matrix[float64] =
  result = xs.map(activation)

proc activation_prime*(xs: Matrix[float64]): Matrix[float64] =
  result = xs.map(activation_prime)

proc activation_last*(xs: Matrix[float64]): Matrix[float64] =
  var sum: float64
  for x in xs.asVector:
     sum += x.exp
  result = xs.map(proc(x: float64): float64 = result = x.exp / sum)

proc `+`*[A: SomeFloat](m: Matrix[A], k: A): Matrix[A] =
  result = m.map(proc(f: A): A = f + k)

proc `+`*[A: SomeFloat](k: A, m: Matrix[A]): Matrix[A] =
  result = m.map(proc(f: A): A = k + f)

proc `-`*[A: SomeFloat](m: Matrix[A], k: A): Matrix[A] =
  result = m.map(proc(f: A): A = f - k)

proc `-`*[A: SomeFloat](k: A, m: Matrix[A]): Matrix[A] =
  result = m.map(proc(f: A): A = k - f)

proc find_if*[T](s: Vector[T], pred: proc(x: T): bool): int =
  result = -1  # return -1 if no items satisfy the predicate
  for i, x in s:
    if pred(x):
      result = i
      break
