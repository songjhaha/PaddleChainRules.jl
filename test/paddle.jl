using PaddleChainRules.Paddle: PaddleModuleWrapper, PaddleFCNet, paddle, to_paddle_tensor, dlpack, pyto_dlpack, pyfrom_dlpack, ispysetup

using Test
using Zygote
import Random
using PyCall
using CUDA
using DLPack
using Functors: fmap

if !ispysetup[]
    return
end

if CUDA.functional()
    @info "using gpu"
else
    @info "using cpu"
end

function compare_grad_wrt_params(model, modelwrap, inputs...)
    pyinputs = fmap(x -> to_paddle_tensor(x,false), inputs)
    paddleoutputs = model(pyinputs...)
    paddlegrad = map(x-> (x.numpy()), paddle.fluid.dygraph.grad(paddleoutputs, model.parameters()))

    grad,  = Zygote.gradient(m->sum(m(inputs...)), modelwrap)
    @test length(paddlegrad) == length(grad.params)
    for i in 1:length(grad.params)
        @test isapprox(sum(paddlegrad[i]), sum(grad.params[i]))
    end
    @test length(grad.params) == length(modelwrap.params)
    @test grad.params[1] !== nothing
    @test grad.params[2] !== nothing
    @test size(grad.params[1]) == size(modelwrap.params[1])
    @test size(grad.params[2]) == size(modelwrap.params[2])
end

function compare_grad_wrt_inputs(model, modelwrap, inputs)
    pyinputs = to_paddle_tensor(inputs,false)
    paddleoutputs = model(pyinputs)
    paddlegrad = map(x-> (x.numpy()), paddle.fluid.dygraph.grad(paddleoutputs, pyinputs))[1]
    grad, = Zygote.gradient(z->sum(modelwrap(z)), inputs)
    @test size(grad) == size(inputs)
    @test length(paddlegrad) == length(grad)
    @test isapprox(sum(paddlegrad), sum(grad))
end

batchsize = 1
indim = 3
outdim = 2
hiddendim = 4

@testset "mlp" begin
    mlp = paddle.nn.Sequential(
        paddle.nn.Linear(indim, hiddendim),
        paddle.nn.Sigmoid(),
        paddle.nn.Linear(hiddendim, outdim)
    )
    mlpwrap = PaddleModuleWrapper(mlp)
    if CUDA.functional()
        mlpwrap = fmap(CUDA.cu, mlpwrap)
    end    
    x = randn(Float32, indim, batchsize)
    if CUDA.functional()
        x = CUDA.cu(x)
    end
    y = mlpwrap(x)
    @test size(y) == (outdim, batchsize)
    compare_grad_wrt_params(mlp, mlpwrap, x)
    compare_grad_wrt_inputs(mlp, mlpwrap, x)
end

@testset "General Net" begin
    NN = paddle.nn.Linear(indim, outdim)
    
    NNwrap = PaddleModuleWrapper(NN)
    if CUDA.functional()
        NNwrap = fmap(CUDA.cu, NNwrap)
    end    
    x = randn(Float32, indim, batchsize)
    if CUDA.functional()
        x = CUDA.cu(x)
    end
    y = NNwrap(x)
    @test size(y) == (outdim, batchsize)
    compare_grad_wrt_params(NN, NNwrap, x)
    compare_grad_wrt_inputs(NN, NNwrap, x)
end

@testset "FC Net" begin
    NN = paddle.nn.Sequential(
        paddle.nn.Linear(indim, hiddendim),
        paddle.nn.Sigmoid(),
        paddle.nn.Linear(hiddendim, outdim)
    )

    NNwrap = PaddleFCNet(indim, outdim, 2, hiddendim; dtype="float32", activation="sigmoid")

    # set same parameters
    pyparams = NN.parameters()
    jlparams = map(x->DLPack.wrap(x, pyto_dlpack), pyparams)
    map((p1,p2)->copy!(p1, p2), NNwrap.params, jlparams)

    if CUDA.functional()
        NNwrap = fmap(CUDA.cu, NNwrap)
    end    
    x = randn(Float32, indim, batchsize)
    if CUDA.functional()
        x = CUDA.cu(x)
    end
    y = NNwrap(x)
    @test size(y) == (outdim, batchsize)
    compare_grad_wrt_params(NN, NNwrap, x)
    compare_grad_wrt_inputs(NN, NNwrap, x)

end

@testset "Error throw" begin
    NN = paddle.nn.Sequential(
        paddle.nn.Conv2D(1,2,5)
    )
    @test_throws ErrorException("Unsupported layer type") PaddleModuleWrapper(NN)
    @test_throws ErrorException("Unsupported activation type") PaddleFCNet(indim, outdim, 2, hiddendim; dtype="float32", activation="celu")
    @test_throws ErrorException("Unsupported data type, use \"float16\", \"float32\" or \"float64\"") PaddleFCNet(indim, outdim, 2, hiddendim; dtype="int8", activation="sigmoid")
end