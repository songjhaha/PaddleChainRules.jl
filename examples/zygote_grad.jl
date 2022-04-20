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

# or use a constructor for full connected network
jlwrap = PaddleFCNet(dim_ins, dim_outs, num_layers, hidden_size; activation="sigmoid")

input = rand(Float32, dim_ins, batch_size)

output = jlwrap(input)

target = rand(Float32, dim_outs, batch_size)
loss(m, x, y) = sum(abs2.(m(x) .- y))

# grad of params 
grad, = Zygote.gradient(m->loss(m, input, target), jlwrap)
# grad of input
grad, = Zygote.gradient(x->loss(jlwrap, x, target), input)



