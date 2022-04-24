# PaddleChainRules

该包的主要思路基本来自于[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)。

该包提供了paddle的全连接神经网络在Julia内的封装，支持Julia Array作为神经网络的输入和输出，通过定义`ChainRulesCore.rrule`使其支持包括Zygote在内的自动微分库的求导操作，也可以和NeuralPDE等结合，作为求解PDE的神经网络后端。

## 使用方法

### 安装依赖

```julia
#install paddlepaddle
using PyCall
run(`$(PyCall.pyprogramname) -m  pip install paddlepaddle`)
## or install the gpu spported version
# run(`$(PyCall.pyprogramname) -m  pip install paddlepaddle-gpu`)
```

### 封装全连接神经网络

支持将只包含`paddle.nn.Linear`和部分激活函数组成的`paddle.nn.Sequential`直接封装：

```julia
using PaddleChainRules.Paddle: paddle, PaddleModuleWrapper, PaddleFCNet
using Zygote

dim_ins = 3
hidden_size = 16
dim_outs = 2
batch_size = 32
num_layers = 2

# now only support full connected Dense network
NN = paddle.nn.Sequential(
        paddle.nn.Linear(dim_ins, hidden_size),
        paddle.nn.Sigmoid(),
        paddle.nn.Linear(hidden_size, dim_outs)
    )

jlwrap = PaddleModuleWrapper(NN)
```

或是直接使用构造函数：

```julia
# or use a constructor for full connected network
jlwrap = PaddleFCNet(dim_ins, dim_outs, num_layers, hidden_size; activation="sigmoid")
```

对于一般的神经网络，也支持直接地封装，但存在一定的[问题]()

### 求导

在Flux以及该包中，都是将输入的`batch_size`维度放在最后一项。

```julia
input = rand(Float32, dim_ins, batch_size)

output = jlwrap(input)

target = rand(Float32, dim_outs, batch_size)
loss(m, x, y) = sum(abs2.(m(x) .- y))

# grad of params 
grad, = Zygote.gradient(m->loss(m, input, target), jlwrap)
# grad of input
grad, = Zygote.gradient(x->loss(jlwrap, x, target), input)
```

### 在GPU上使用

首先确认安装的是paddlepaddle-gpu的版本。

```julia
using PaddleChainRules.Paddle: paddle, PaddleModuleWrapper, PaddleFCNet
using CUDA
using Zygote
# paddle-gpu will use cuda defualtly if cuda is useable
# or set up the device by hand
paddle.device.set_device("gpu:0")

dim_ins = 3
hidden_size = 16
dim_outs = 2
batch_size = 32
num_layers = 2

# now only support full connected Dense network
NN = paddle.nn.Sequential(
        paddle.nn.Linear(dim_ins, hidden_size),
        paddle.nn.Sigmoid(),
        paddle.nn.Linear(hidden_size, dim_outs)
    )

jlwrap = PaddleModuleWrapper(NN)

# or use a constructor for full connected network
jlwrap = PaddleFCNet(dim_ins, dim_outs, num_layers, hidden_size; activation="sigmoid")

# make sure the inputs and outputs are the CUDA Arrays
input = CUDA.cu(rand(Float32, dim_ins, batch_size))

output = jlwrap(input)

target = CUDA.cu(rand(Float32, dim_outs, batch_size))
loss(m, x, y) = sum(abs2.(m(x) .- y))

# grad of params 
grad, = Zygote.gradient(m->loss(m, input, target), jlwrap)
# grad of input
grad, = Zygote.gradient(x->loss(jlwrap, x, target), input)
```

### NeuralPDE

基本只需要修改两处代码，一个是把神经网络修改成PaddleModuleWrapper，另一处是使用`Optimisers.destructure()`得到Flatten的神经网络的参数。具体代码见[example](examples/demo_neuralpde.jl)。



## 实现方法

[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)封装了Python中的Torch和Jax的神经网络。这里对Paddle的封装基本上模仿其对Torch的封装（也存在一些不同）。

1. 为了实现与Julia生态的对接，需要满足对输入和输出皆是julia上的Array的支持，[DLPack.jl](https://github.com/pabloferz/DLPack.jl)实现了tensor和Julia array之间交换数据的协议，也支持GPU上的数据，因此可以对输入输出都调用DLPack进行转换，对于二维以上的tensor，转换后array的维度会反转。

   ```julia
   using PyCall
   using DLPack
   dlpack = pyimport("paddle.utils.dlpack")
   paddle = pyimport("paddle")
   pyto_dlpack(x) = @pycall dlpack.to_dlpack(x)::PyObject
   pyfrom_dlpack(x) = @pycall dlpack.from_dlpack(x)::PyObject
   
   # wrap a paddle tensor
   pyv = paddle.randn((10,2))
   jlv = DLPack.wrap(pyv, pyto_dlpack)
   # size(jlv) == (2,10)
   
   # share Julia arrays to python
   pyv = DLPack.share(jlv, PyObject, pyfrom_dlpack)
   ```

   

2. 将神经网络分成无状态的网络模型和参数两部分。这么做的好处是可以方便地调用`Optimisers.destructure(modelwrapper)`获取参数和分离模型，以及使用julia生态的优化器对参数进行更新。

   ```julia
   struct PaddleModuleWrapper
       NN::PaddleStatelessModule
       dtype::Type
       params::Vector
   end
   
   Base.length(f::PaddleModuleWrapper) = length(f.params)
   Base.iterate(f::PaddleModuleWrapper) = iterate(f.params)
   Base.iterate(f::PaddleModuleWrapper, state) = iterate(f.params, state)
   
   @functor PaddleModuleWrapper (params,)
   
   # forward
   function (wrap::PaddleModuleWrapper)(args...; kwargs...)
       out = wrap.NN(map(x -> DLPack.share(x, PyObject, pyfrom_dlpack), wrap.params), fmap(x -> DLPack.share(x, PyObject, pyfrom_dlpack), args)...)
       res = fmap(x->DLPack.wrap(x, pyto_dlpack), out)
       return res
   end
   ```

   在[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)中，使用了[functorch](https://github.com/pytorch/functorch)库作为分离模型和参数的方法。Paddle没有相关的实现，但明显地手动实现一个无状态的网络并不复杂，只需要定义模型的`forward(params, x)`方法即可。对于全连接神经网络，可以分别实现`Linear layer`和`activation layer`：

   ```julia
   abstract type PaddleStatelessModule end
   abstract type PaddleStatelessLayer end
   
   function (stateless_module::PaddleStatelessFCNet)(params::Vector, inputs; kwinputs...)
       out = PyNULL()
       copy!(out, inputs)
       state = 1
       for layer in stateless_module.layers
           state = PaddleLayerForward!(out, params, state, layer)
       end
       return out
   end
   
   # wrap paddle's layer
   struct PaddleLinear<:PaddleStatelessLayer
       features_ins::Int
       features_outs::Int
   end
   
   function PaddleLayerForward!(out::PyObject, params::Vector, state::Int, L::PaddleLinear)
       weight, state = iterate(params, state)
       bias, state = iterate(params, state)
       copy!(out, paddle.matmul(out, weight))
       copy!(out, paddle.add(out, bias))
       return state
   end
   
   struct PaddleActivation<:PaddleStatelessLayer
       act::PyObject
   end
   
   function PaddleLayerForward!(out::PyObject, params::Vector, state::Int, L::PaddleActivation)
       copy!(out, L.act(out))
       return state
   end
   ```

   对于更一般的模型，暂时的解决方法是在每次调用forward之前，通过`tensor.set_value()`方法设定params的值：

   ```julia
   # a rough solution for General Net
   struct PaddleStatelessGeneralNet&lt;:PaddleStatelessModule
       NN::PyObject
   end
   
   function (stateless_module::PaddleStatelessGeneralNet)(params::Vector, inputs; kwinputs...)
       map((p,p_new) -> p.set_value(p_new), stateless_module.NN.parameters(), params)
       out = stateless_module.NN(inputs)
       return out
   end
   ```

   

3. 定义`ChainRulesCore.rrule`规则，这部分的实现基本上和[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)上的相同，主要是注意对tensor和array的转换：

   ```julia
   function ChainRulesCore.rrule(wrap::PaddleModuleWrapper, args...; kwargs...)
       T = typeof(first(wrap.params))
       pyparams = fmap(x -> to_paddle_tensor(x,false), wrap.params)
       pyargs = fmap(x -> to_paddle_tensor(x,false), args)
       paddle_primal, vjp_func = vjp(wrap.NN, pyparams, pyargs...; kwargs...)
       project = ProjectTo(args)
       function pullback(Δ)
           cΔ = fmap(x->Adapt.adapt(PyAdaptor{T}(), x), Δ)
           pycΔ = fmap(x->DLPack.share(x, PyObject, pyfrom_dlpack), cΔ)
           paddle_tangent_vals = vjp_func(pycΔ)
           jlparams_tangents = map(x -> DLPack.wrap(x, pyto_dlpack), paddle_tangent_vals[1])
           args_tangents = project(fmap(x -> DLPack.wrap(x, pyto_dlpack), paddle_tangent_vals[2]))
           return (Tangent{PaddleModuleWrapper}(; NN = NoTangent(), dtype = NoTangent(), params = jlparams_tangents), args_tangents...)
       end
       res = fmap(x->DLPack.wrap(x, pyto_dlpack), paddle_primal)
       return res, pullback
   end
   ```

   与[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)的不同是需要自己实现类似Jax的`vjp`函数：

   ```julia
   function vjp(stateless_module::PaddleStatelessFCNet, pyparams::Vector, pyargs...; kwargs...) # grad wrt params and args
       res = stateless_module(pyparams, pyargs...; kwargs...)
       paramslen = length(pyparams)
       function vjp_func(Δ)
           grad = paddle.fluid.dygraph.grad(res, vcat(pyparams, pyargs...), Δ, retain_graph=true)
           return (grad[1:paramslen], grad[paramslen+1:end])
       end
       return res, vjp_func
   end
   ```

4. 另外还实现了全连接神经网络的构造函数：

   ```julia
   PaddleFCNet(dim_ins, dim_outs, num_layers, hidden_size; dtype="float32", activation="sigmoid")
   ```

   

## 性能测试

1. 比较前向传播和反向传播时，在julia上的封装和使用paddle原生api的效率的差异，这两组分别记为`jl`和`paddle`，同时引入另一组对照组`funcpaddle`，使用的是这里实现的`PaddleStatelessModule`来计算前向传播，在计算梯度时，`paddle`和`funcpaddle`使用`paddle.grad()`方法，而`jl`使用`Zygote.gradient()`。实验固定了神经网络的结构和参数的数量，采用的模型是` paddle.nn.Sequential(paddle.nn.Linear(indim, hiddendim), paddle.nn.ReLU(), paddle.nn.Linear(hiddendim, outdim))`，其中`indim=8`，`outdim=4`，`hiddendim=64`，`batch size`分别使用`[1, 8, 16, 32, 64]`进行实验。`paddle`和`funcpaddle`的输入使用固定的tensor，而`jl`使用固定的array。每个实验分别进行100次，下表记录了使用[BenchMarkTools](https://github.com/JuliaCI/BenchmarkTools.jl)估计的最小的运行时间：

   **Forward:**

   | Batch Size |     jl     |   paddle   | funcpaddle |
   | :--------: | :--------: | :--------: | :--------: |
   |     1      | 452.917 μs | 143.925 μs | 161.252 μs |
   |     8      | 435.753 μs | 150.479 μs | 160.765 μs |
   |     16     | 424.979 μs | 146.872 μs | 172.635 μs |
   |     32     | 448.490 μs | 156.428 μs | 183.635 μs |
   |     64     | 451.121 μs | 169.250 μs | 187.323 μs |

   **Backward:**

   | Batch Size |    jl    |   paddle   | funcpaddle |
   | :--------: | :------: | :--------: | :--------: |
   |     1      | 1.025 ms | 602.706 μs | 525.320 μs |
   |     8      | 1.151 ms | 630.728 μs | 540.370 μs |
   |     16     | 1.092 ms | 627.285 μs | 534.335 μs |
   |     32     | 1.153 ms | 598.154 μs | 573.360 μs |
   |     64     | 1.178 ms | 659.823 μs | 605.877 μs |

   实验的代码可以在colab上查看：[Benchmark of Forward and Backward](https://colab.research.google.com/drive/13cz63OcGecnI1kUi5jzNYPxFKwpTW6lW?usp=sharing)，在gpu上的测试结果与cpu上相似，因此不在这里列出。

   可以看到`jl`在前向传播时的运行时间几乎是paddle原生的三倍，而在反向传播时几乎是paddle原生的两倍。这部分的差异主要是定义`ChainRulesCore.rrule`时产生的额外开销，如对输入和输出以及模型参数的转换，或是调用`paddle.fluid.dygraph.grad(res, vcat(pyparams, pyargs...), Δ, retain_graph=true)`时对输入的求导，和保存计算图等，也有一部分是运行`Zygote`额外的开销。



2. 在NeuralPDE的例子中，测试了[DiffEqFlux](https://github.com/SciML/DiffEqFlux.jl)，[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl) 和该包的实现，在求解2d的泊松方程时的效率差异。实验固定了泊松方程的参数，神经网络的结构，训练时的采样算法和迭代的次数。神经网络固定为三层的全连接神经网络，设定不同的`hidden layer`的大小。实验的结果如下表：

   | hiddendims | numbers of params | DiffEqFlux | Torch in PyCallChainRules | PaddleChianRules |
   | :--------: | :---------------: | :--------: | :-----------------------: | :--------------: |
   |     8      |        105        | 43.039 ms  |        290.360 ms         |    309.928 ms    |
   |     16     |        337        | 55.741 ms  |        310.712 ms         |    318.576 ms    |
   |     32     |       1185        | 80.820 ms  |        311.391 ms         |    336.273 ms    |

   实验的代码可以在colab上查看：[Benchmark of NeuralPDE](https://colab.research.google.com/drive/1gK5dZ1k6zrH-KIQn-wJ1gfwenIxhxn32?usp=sharing)

   `PyCallChainRules`和`PaddleChianRules`明显慢于`DiffEqFlux`，他们两个的运行效率差不多，也是因为`PaddleChianRules`采用了前者的实现方案。当模型的参数增加时，`PyCallChainRules`和`PaddleChianRules`的运行效率并没有明显的降低，可能是由于在这个实验中，参数的大小并非他们的计算瓶颈，而来自于额外的开销。

## 未来可能的优化方向

1. 对一般的神经网络结构更好的支持。可以参考[functorch](https://github.com/pytorch/functorch)上的实现方案。
2. 由于只定义了rrule，和为了更一般的使用场景，每次求导都需要对参数和输入都进行求导，当输入的维度和数量不是太大时，这种额外的开销也许影响不大。在某些情况可以禁用对输入的求导，只需要修改`ChainRulesCore.rrule`和`vjp`的实现。
3. 在更新参数时，目前的NeuralPDE使用的是`Optimisers.update!(opt, params, grad)`，可以正常的运行，但在其他的一些使用场景下可能存在问题，详见https://github.com/rejuvyesh/PyCallChainRules.jl/issues/19

