# PaddleChainRules

The idea is from [PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)

a small demo package of wrapping a full cennected Dense network of [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) in julia, and make it differentiable by `ChainRulesCore.rrule`.

## Example
```julia
#install paddlepaddle
using PyCall
run(`$(PyCall.pyprogramname) -m  pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html`)

using PaddleChainRules.Paddle: paddle, PaddleModuleWrapper
using Zygote

dim_ins = 3
hidden_size = 16
dim_outs = 2
batch_size = 32

# now only support full connected Dense network
NN = paddle.nn.Sequential(
        paddle.nn.Linear(dim_ins, hidden_size),
        paddle.nn.Sigmoid(),
        paddle.nn.Linear(hidden_size, dim_outs)
    )

jlwrap = PaddleModuleWrapper(NN)

input = rand(Float32, dim_ins, batch_size)

output = jlwrap(input)

target = rand(Float32, dim_outs, batch_size)
loss(m, x, y) = sum(abs2.(m(x) .- y))

# grad of params 
grad, = Zygote.gradient(m->loss(m, input, target), jlwrap)
# grad of input
grad, = Zygote.gradient(x->loss(jlwrap, x, target), input)
```

And there is a [demo](examples/demo_neuralpde.jl) for neuralPDE.

## TODO
- In the demo of neuralPDE, this package is much slower than [PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl) and Flux.jl, need to imporve the speed.
- Now only the Dense network is supported, more genneral network structure?
- test code. compare output of forwrad and backward to the result from paddle's api
- GPU support.
- Some benchmarks:
    + forward and backward
    + possion equation with NeuralPDE, compared with PyCallChainRules, Flux and [PaddleScience](https://github.com/PaddlePaddle/PaddleScience)




