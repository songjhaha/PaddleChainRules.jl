module PaddleChainRules

using DLPack
using Requires

import FillArrays
import Adapt
import Functors: fmap
import ChainRulesCore

struct PyAdaptor{T} end
Adapt.adapt_storage(to::PyAdaptor{T}, x::AbstractArray) where {T} = convert(Array, x)
Adapt.adapt_storage(to::PyAdaptor{T}, x::StridedArray) where {T} = x
Adapt.adapt_storage(to::PyAdaptor{<:AbstractArray}, x::FillArrays.AbstractFill) = collect(x)
# Handle views
Adapt.adapt_structure(to::PyAdaptor{<:AbstractArray}, x::A) where {A <: SubArray}  = collect(x)

### XXX: what's a little piracy between us
fmap(f, x::ChainRulesCore.Tangent) = fmap(f, x.backing)


include("paddle.jl")




end # module
