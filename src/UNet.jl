module UNet

export Unet

using StatsBase, Flux, Random
using Flux: @functor

using Distributions: Normal

testing_seed = nothing

test_mode() = (global testing_seed = 1)
run_mode() = (global testing_seed = nothing)

include("utils.jl")
include("model.jl")

end # module
