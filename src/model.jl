function BatchNormWrap(out_ch::Integer)
    Chain(x->expand_dims(x,2), BatchNorm(out_ch), x->squeeze(x))
end

function UNetConvBlock(in_chs::Integer, out_chs::Integer, kernel=3, activation=relu; ndim=2)
  kernel = ntuple(_ -> kernel, ndim)
  pad = div.(kernel, 2)
  return Chain(Conv(kernel, in_chs=>out_chs, activation, pad=pad; init=_random_normal))
end

function ConvDown(in_chs::Integer, out_chs::Integer, kernel=4; ndim=2)
  kernel = ntuple(_ -> kernel, ndim)
  stride = ntuple(_ -> 2, ndim)
  block = Chain(Conv(kernel, in_chs=>out_chs, pad=SamePad(); init=_random_normal), MaxPool(stride; pad=0))
  return block
end

struct UNetUpBlock
  conv
  up
end

@functor UNetUpBlock

function UNetUpBlock(in_chs::Integer, out_chs::Integer; kernel=3, ndim=2)
    stride = ntuple(_ -> 2, ndim)
    kernel = ntuple(_ -> kernel, ndim)
    up = Upsample(:bilinear, scale=stride)
    return UNetUpBlock(Conv(kernel, in_chs=>out_chs, pad=SamePad(); init=_random_normal), up)
end

function (u::UNetUpBlock)(x::AbstractArray{T, N}, bridge::AbstractArray{T, N}) where {T, N}
  x = u.up(u.conv(x))
  return cat(x, bridge, dims=N-1)
end

"""
    Unet(channels::Int = 1, labels::Int = channels)

  Initializes a [UNet](https://arxiv.org/pdf/1505.04597.pdf) instance with the given number of `channels`, typically equal to the number of channels in the input images.
  `labels`, equal to the number of input channels by default, specifies the number of output channels.
"""
struct Unet{D}
  conv_down_blocks
  init_conv_block
  conv_blocks::NTuple{D, <:Chain}
  up_blocks
  out_blocks
  func
end

@functor Unet


function Unet(channels::Integer = 1, labels::Int = channels, depth::Integer=5; ndim=2, pad_input=true)
  if depth < 3
    @warn "not recommended to use less than 3 levels"
  end
  init_conv_block = channels >= 3 ? UNetConvBlock(channels, 64; ndim=ndim) : Chain(UNetConvBlock(channels, 3; ndim=ndim), UNetConvBlock(3, 64; ndim=ndim))

  conv_down_blocks = tuple([i == depth ? x -> x : ConvDown(2^(5+i), 2^(5+i); ndim=ndim) for i=1:depth]...)

  conv_blocks = tuple([UNetConvBlock(2^(5+i), 2^(6+min(i, depth-1)); ndim=ndim) for i=1:depth]...)

  up_blocks = tuple([UNetUpBlock(2^(6+min(i+1, depth-1)), 2^(5+i); ndim=ndim) for i=depth-1:-1:1]...)

  out_blocks = Chain(x -> leakyrelu.(x, 0.2f0), Conv(ntuple(_->1, ndim), 128=>labels; init=_random_normal), x -> tanh.(x))

  func = pad_input ? _padded_unet : _unet

  return Unet{depth}(conv_down_blocks, init_conv_block, conv_blocks, up_blocks, out_blocks, func)
end


(u::Unet{D})(x::AbstractArray{T, N}) where {D, T, N} = u.func(u, x)

function _padded_unet(u::Unet{D}, x::AbstractArray{T, N}) where {D, T, N}
  xp = padz_unet(x, D-1)
  ux = _unet(u, xp)
  return crop_unet(ux, size(x))
end

function _unet(u::Unet{D}, x::AbstractArray{T, N}) where {D, T, N}
  xcis = (u.init_conv_block(x),)

  for d=1:D
    xcis = (xcis..., u.conv_blocks[d](u.conv_down_blocks[d](xcis[d])))
  end

  ux = u.up_blocks[1](xcis[D+1], xcis[D-1])

  for d=2:D-1
    ux = u.up_blocks[d](ux, xcis[D-d])
  end

  return u.out_blocks(ux)
end

