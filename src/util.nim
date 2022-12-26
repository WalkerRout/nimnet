
import neo
import math
import sequtils

proc activation*(x: float64): float64 =
  result = 1 / (1 + (-x).exp)

proc activation*(xs: Matrix[float64]): Matrix[float64] =
  result = xs.map(activation)

proc activation_prime*(x: float64): float64 =
  result = activation(x) * (1 - activation(x))

proc activation_prime*(xs: Matrix[float64]): Matrix[float64] =
  result = xs.map(activation_prime)

proc `+`*[A: SomeFloat](m: Matrix[A], k: A): Matrix[A] =
  result = m.map(proc(f: A): A = f + k)

proc `+`*[A: SomeFloat](k: A, m: Matrix[A]): Matrix[A] =
  result = m.map(proc(f: A): A = k + f)

proc `-`*[A: SomeFloat](m: Matrix[A], k: A): Matrix[A] =
  result = m.map(proc(f: A): A = f - k)

proc `-`*[A: SomeFloat](k: A, m: Matrix[A]): Matrix[A] =
  result = m.map(proc(f: A): A = k - f)
