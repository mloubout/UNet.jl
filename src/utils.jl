import ChainRulesCore: rrule, NoTangent

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

function padz_unet(x::AbstractArray{T, 4}, D::Integer) where T
    n1, n2, n3, n4 = size(x)
    p1 = mod(-n1, 2^D)
    p2 = mod(-n2, 2^D)
    px = fill!(similar(x, (n1+p1, n2+p2, n3, n4)), zero(T))
    px[p1+1:end, p2+1:end, :, :] .= x
    return px
end

function padz_unet(x::AbstractArray{T, 4}, N::NTuple{4, Integer}) where T
    px = fill!(similar(x, N), zero(T))
    c1, c2 = size(x)[1:2] .- N[1:2] .+ 1
    px[c1:end, c2:end, :, :] .= x
    return px
end

function crop_unet(x::AbstractArray{T, 4}, N::NTuple{4, Integer}) where T
    c1, c2 = size(x)[1:2] .- N[1:2] .+ 1
    return x[c1:end, c2:end, :, :]
end

function rrule(::typeof(padz_unet), x::DenseArray{T, 4}, D::Integer) where T
    pz = padz_unet(x, D)
    bck(dy) = (NoTangent(), crop_unet(dy, size(x)), NoTangent())
    return pz, bck
end

function rrule(::typeof(crop_unet), x::DenseArray{T, 4}, N::NTuple{4, Integer}) where T
    cz = crop_unet(x, N)
    bck(dy) = (NoTangent(), padz_unet(dy, size(x)), NoTangent())
    return cz, bck
end
