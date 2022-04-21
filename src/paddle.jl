module Paddle

using PyCall

using ChainRulesCore
using DLPack
using Functors: @functor
using Adapt

using ..PaddleChainRules: PyAdaptor, fmap

const dlpack = PyNULL()
const paddle = PyNULL()
const ispysetup = Ref{Bool}(false)

pyto_dlpack(x) = @pycall dlpack.to_dlpack(x)::PyObject
pyfrom_dlpack(x) = @pycall dlpack.from_dlpack(x)::PyObject

include("Net.jl") 

struct PaddleModuleWrapper
    NN::PaddleStatelessModule
    dtype::Type
    params::Vector
end

Base.show(io::IO, f::PaddleModuleWrapper) = print(io, f.NN, " ", f.dtype)

Base.length(f::PaddleModuleWrapper) = length(f.params)
Base.iterate(f::PaddleModuleWrapper) = iterate(f.params)
Base.iterate(f::PaddleModuleWrapper, state) = iterate(f.params, state)

function PaddleModuleWrapper(paddle_module)
    params = paddle_module.parameters()
    jlparams = map(x->DLPack.wrap(x, pyto_dlpack), params)
    dtype = eltype(jlparams[1])
    return PaddleModuleWrapper(PaddleStatelessModule(paddle_module), dtype, jlparams)
end

@functor PaddleModuleWrapper (params,)

function (wrap::PaddleModuleWrapper)(args...; kwargs...)
    out = wrap.NN(map(x -> DLPack.share(x, PyObject, pyfrom_dlpack), wrap.params), fmap(x -> DLPack.share(x, PyObject, pyfrom_dlpack), args)...)
    res = fmap(x->DLPack.wrap(x, pyto_dlpack), out)
    return res
end

function vjp(stateless_module::PaddleStatelessFCNet, pyparams::Vector, pyargs...; kwargs...) # grad wrt params and args
    res = stateless_module(pyparams, pyargs...; kwargs...)
    paramslen = length(pyparams)
    function vjp_func(Δ)
        # compute the grad of both params and args? will it cost more during training?
        grad = paddle.fluid.dygraph.grad(res, vcat(pyparams, pyargs...), Δ, retain_graph=true)
        return (grad[1:paramslen], grad[paramslen+1:end])
    end
    return res, vjp_func
end

function vjp(stateless_module::PaddleStatelessGeneralNet, pyparams::Vector, pyargs...; kwargs...) # grad wrt params and args
    res = stateless_module(pyparams, pyargs...; kwargs...)
    paramslen = length(pyparams)
    function vjp_func(Δ)
        # compute the grad of both params and args? will it cost more during training?
        grad = paddle.fluid.dygraph.grad(res, vcat(stateless_module.NN.parameters(), pyargs...), Δ, retain_graph=true)
        return (grad[1:paramslen], grad[paramslen+1:end])
    end
    return res, vjp_func
end

## grad wrt params only
# function vjp_wrt_params(stateless_module::PaddleStatelessModule, pyparams::Vector, pyargs...; kwargs...)
#     res = stateless_module(pyparams, pyargs...; kwargs...)
#     function vjp_func(Δ)
#         # compute the grad of params
#         grad = paddle.fluid.dygraph.grad(res, pyparams, Δ, retain_graph=true)
#         return grad
#     end
#     return res, vjp_func
# end

function to_paddle_tensor(x::AbstractArray, stop_grad::Bool)
    out = DLPack.share(x, PyObject, pyfrom_dlpack)
    out.stop_gradient = stop_grad
    return out
end

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


function __init__()
    try
        copy!(paddle, pyimport("paddle"))
        copy!(dlpack, pyimport("paddle.utils.dlpack"))
        ispysetup[] = true     
    catch err
        @warn """PaddleChainRules.jl has failed to import paddle from Python.
                 Please make sure these are installed. 
        """
        @debug err
        ispysetup[] = false  
    end
end


end
