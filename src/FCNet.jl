abstract type PaddleStatelessModule end
abstract type PaddleStatelessLayer end

struct PaddleStatelessFCNet<:PaddleStatelessModule
    layers::Vector{PaddleStatelessLayer}
end

function (stateless_module::PaddleStatelessFCNet)(params::Vector, inputs; kwinputs...)
    out = PyNULL()
    copy!(out, inputs)
    state = 1
    for layer in stateless_module.layers
        state = PaddleLayerForward!(out, params, state, layer)
    end
    return out
end

function PaddleStatelessModule(paddle_module)
    pybuiltin("isinstance")(paddle_module, paddle.nn.Layer) || error("Not a paddle.nn.Layer")
    layers = PaddleStatelessLayer[]
    for layer in paddle_module
        if pybuiltin("isinstance")(layer, paddle.nn.Linear)
            push!(layers, PaddleLinear(layer.weight.shape...))
        else 
            push!(layers, PaddleActivation(layer))
        end
    end
    paddel_fc_net = PaddleStatelessFCNet(layers)
    return paddel_fc_net
end

# wrap paddle's layer
struct PaddleLinear<:PaddleStatelessLayer
    features_ins::Int
    features_outs::Int
end

Base.show(io::IO, f::PaddleLinear) = print(io, "Linear($(f.features_ins), $(f.features_outs))")

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

Base.show(io::IO, f::PaddleActivation) = print(io, "$(f.act)")

function PaddleLayerForward!(out::PyObject, params::Vector, state::Int, L::PaddleActivation)
    copy!(out, L.act(out))
    return state
end




