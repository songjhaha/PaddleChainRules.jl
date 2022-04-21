abstract type PaddleStatelessModule end
abstract type PaddleStatelessLayer end

# a rough solution for General Net
struct PaddleStatelessGeneralNet<:PaddleStatelessModule
    NN::PyObject
end

function (stateless_module::PaddleStatelessGeneralNet)(params::Vector, inputs; kwinputs...)
    map((p,p_new)->p.set_value(p_new), stateless_module.NN.parameters(), params)
    out = stateless_module.NN(inputs)
    return out
end

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
    if pybuiltin("isinstance")(paddle_module, paddle.nn.Sequential)
        layers = PaddleStatelessLayer[]
        for layer in paddle_module
            if pybuiltin("isinstance")(layer, paddle.nn.Linear)
                push!(layers, PaddleLinear(layer.weight.shape...))
            elseif pybuiltin("isinstance")(layer, (paddle.nn.Sigmoid, paddle.nn.ReLU, paddle.nn.Tanh))
                push!(layers, PaddleActivation(layer))
            else
                throw(error("Unsupported layer type"))
            end
        end
        paddel_fc_net = PaddleStatelessFCNet(layers)
        return paddel_fc_net
    else
        paddle_net = PaddleStatelessGeneralNet(paddle_module)
        return paddle_net
    end
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

# easy constructor for full connected network
function PaddleFCNet(dim_ins, dim_outs, num_layers, hidden_size; dtype="float32", activation="sigmoid")
    act = PyNULL()
    if activation == "sigmoid"
        copy!(act, paddle.nn.Sigmoid())
    elseif activation == "tanh"
        copy!(act, paddle.nn.Tanh())
    elseif activation == "relu"
        copy!(act, paddle.nn.ReLU())
    else
        throw(error("Unsupported activation type"))
    end
    
    if !(dtype in ["float16", "float32", "float64"])
        throw(error("Unsupported data type, use \"float16\", \"float32\" or \"float64\""))
    end

    pyparams = PyObject[]
    layers = PaddleStatelessLayer[]
    lsize, rsize = 0, 0
    for i in 1:num_layers
        if i == 1
            lsize = dim_ins
            rsize = hidden_size
        elseif i == num_layers
            lsize = hidden_size
            rsize = dim_outs
        else
            lsize = hidden_size
            rsize = hidden_size
        end
        push!(pyparams, paddle.static.create_parameter(shape=PyVector([lsize, rsize]), dtype=dtype))
        push!(pyparams, paddle.static.create_parameter(shape=PyVector([rsize]), dtype=dtype))
        push!(layers, PaddleLinear(lsize, rsize))
        if i != num_layers
            push!(layers, PaddleActivation(act))
        end
    end
    jlparams = map(x->DLPack.wrap(x, pyto_dlpack), pyparams)
    dtype = eltype(jlparams[1])
    paddel_fc_net = PaddleStatelessFCNet(layers)
    return PaddleModuleWrapper(paddel_fc_net, dtype, jlparams)
end


