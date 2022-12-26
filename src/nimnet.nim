
import math
import neo

import util

type
  Layer = object of RootObj
    W: Matrix[float64]
    b: Matrix[float64]
    z: Matrix[float64]
    a: Matrix[float64]
    dW: Matrix[float64]
    db: Matrix[float64]
    dz: Matrix[float64]
  Network = object of RootObj
    L: int
    n: seq[int]
    layers: seq[Layer]
    e: Matrix[float64]

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
  for l in countup(1, nn.L):
    var curr = addr nn.layers[l]
    curr.z = (curr.W * nn.layers[l - 1].a) + curr.b
    curr.a = activation(curr.z)

proc error(nn: var Network, y: Matrix[float64]) =
  var last = addr nn.layers[nn.L]
  nn.e = -1.0 * (y * last.a.ln) + (1.0 - y) * (1.0 - last.a).ln

proc delta(nn: var Network, y: Matrix[float64]) =
  let last = addr nn.layers[nn.L]

  last.dz = last.a - y
  last.dW = last.dz * nn.layers[nn.L - 1].a.t
  last.db = last.dz

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

proc fit(nn: var Network, Xs, Ys: Matrix[float64], epochs: int = 100, alpha: float64 = 0.01) =
  for epoch in countup(0, epochs-1):
    var c = 0.0
    var n_c = 0

    for i in countup(0, Xs.dim[0] - 1):
      let x = Xs.row(i).asMatrix(Xs.dim[1], 1)
      let y = Ys.row(i).asMatrix(Ys.dim[1], 1)

      nn.forward(x)
      nn.error(y)
      nn.delta(y)
      nn.gradient_descent(alpha)

      c += nn.e[0, 0]

      let y_pred = nn.predict(x)
      let res = y_pred[0, 0] > 0.5
      if res == y[0, 0].bool:
        n_c += 1

    echo "Iteration: ", epoch
    echo "Accuracy:", (n_c / Xs.dim[0]) * 100.0

proc main() =
  let xs = matrix(@[
    @[0.0,   0.0],
    @[0.0,   100.0],
    @[100.0, 0.0],
    @[100.0, 100.0],
    @[100.0, 100.0],
  ])

  let ys = matrix(@[
    @[0.0],
    @[1.0],
    @[1.0],
    @[0.0],
    @[0.0],
  ])

  var nn = network(@[2, 24, 1])
  nn.fit(xs, ys, 1000, 0.01)

  let sample = matrix(@[@[0.0, 100.0]]).t
  let threshold = 0.5
  echo nn.predict(sample), " => ", (nn.predict(sample) + 1 - threshold).floor

when isMainModule:
  main()

