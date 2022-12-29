
import math
import neo

import std/parsecsv
import std/strutils

import util

type
  Layer = object of RootObj
    W:  Matrix[float64]
    b:  Matrix[float64]
    z:  Matrix[float64]
    a:  Matrix[float64]
    dW: Matrix[float64]
    db: Matrix[float64]
    dz: Matrix[float64]

  Network = object of RootObj
    L: int
    n: seq[int]
    e: Matrix[float64]
    layers: seq[Layer]

proc network(architecture: seq[int]): Network =
  result.L = architecture.len - 1
  result.n = architecture
  newSeq(result.layers, result.L+1)

  for i in countup(1, result.L):
    var layer: Layer
    layer.W = randomMatrix(result.n[i], result.n[i - 1])
    layer.b = randomMatrix(result.n[i], 1)
    layer.z = ones(result.n[i], 1)
    layer.a = ones(result.n[i], 1)

    layer.dW = ones(result.n[i], result.n[i - 1])
    layer.db = ones(result.n[i], 1)
    layer.dz = ones(result.n[i], 1)

    result.layers[i] = layer

  result.layers[0].a = ones(result.n[0], 1) # will change
  result.e = ones(1, 1)

proc forward(nn: var Network, X: Matrix[float64]) = 
  nn.layers[0].a = X

  for l in countup(1, nn.L - 1):
    var curr = addr nn.layers[l]
    curr.z = (curr.W * nn.layers[l - 1].a) + curr.b
    curr.a = activation(curr.z)

  var curr = addr nn.layers[nn.L]
  curr.z = (curr.W * nn.layers[nn.L - 1].a) + curr.b
  curr.a = activation_last(curr.z)

proc error(nn: var Network, y: Matrix[float64]) =
  var last = addr nn.layers[nn.L]
  let index = find_if[float64](y.asVector, proc(x: float64): bool = result = x == 1.0)
  nn.e = matrix(@[@[-1.0 * last.a.asVector[index].ln]])

proc delta(nn: var Network, y: Matrix[float64]) =
  var last = addr nn.layers[nn.L]

  last.dz = last.a - y
  last.dW = last.dz * nn.layers[nn.L - 1].a.t
  last.db = last.dz # works out to just be last.dz for a single sample

  for l in countdown(nn.L - 1, 1):
    let curr = addr nn.layers[l]
    curr.dz = (nn.layers[l + 1].W.t * nn.layers[l + 1].dz) |*| activation_prime(curr.z)
    curr.dW = curr.dz * nn.layers[l - 1].a.t
    curr.db = curr.dz

proc gradient_descent(nn: var Network, alpha: float64) =
  for l in countup(1, nn.L):
    let curr = addr nn.layers[l]
    curr.W -= alpha * curr.dW
    curr.b -= alpha * curr.db

proc predict(nn: var Network, X: Matrix[float64]): Matrix[float64] =
  nn.forward(X)
  result = nn.layers[nn.L].a

proc fit(nn: var Network, Xs, Ys: Matrix[float64], epochs: int = 500, alpha: float64 = 0.05): float64 =
  for epoch in countup(0, epochs-1):
    var c = 0

    for i in countup(0, Xs.dim[0] - 1):
      let x = Xs.row(i).asMatrix(Xs.dim[1], 1)
      let y = Ys.row(i).asMatrix(Ys.dim[1], 1)

      nn.forward(x)
      nn.error(y)
      nn.delta(y)
      nn.gradient_descent(alpha)

      let y_pred = nn.predict(x)
      let res = y_pred[0, 0] > 0.5
      if res == y[0, 0].bool:
        c += 1

    # accuracy
    result = c / Xs.dim[0]

proc main() =
  # specific invariant required;
  # single newline at end of each file
  let xs = read_matrix("Xs.csv")
  let ys = read_matrix("Ys.csv")

  var nn = network(@[xs.row(0).len, 4, ys.row(0).len])
  let accuracy = nn.fit(xs, ys, alpha=0.1)

  let row = 2
  let sample = xs.row(row).asMatrix(xs.dim[1], 1)

  echo "Training accuracy ", accuracy * 100, "%"
  echo "Probabilities: ", nn.predict(sample).t
  echo "Predicted result: ", (nn.predict(sample) + 0.5).floor.t.asVector
  echo "Expected result: ", ys.row(row)

when isMainModule:
  main()
