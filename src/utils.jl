export bce, loss_bce

function _random_normal(shape...)
  isnothing(testing_seed) ?  Float32.(rand(Normal(0.f0, 0.02f0),shape...)) : ones(Float32, shape...)
end

expand_dims(x, n::Int) = reshape(x, ones(Int64, n)..., size(x)...)

function squeeze(x) 
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)..., 1)
    end
end

function bce(ŷ, y; ϵ=gpu(fill(eps(first(ŷ)), size(ŷ)...)))
  l1 = -y.*log.(ŷ .+ ϵ)
  l2 = (1 .- y).*log.(1 .- ŷ .+ ϵ)
  return l1 .- l2
end

function loss_bce(x, y)
  op = clamp.(u(x), 0.001f0, 1.f0)
  return mean(bce(op, y))
end
