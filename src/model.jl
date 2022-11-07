function BatchNormWrap(out_ch::Integer)
    Chain(x->expand_dims(x,2), BatchNorm(out_ch), x->squeeze(x))
end

UNetConvBlock(in_chs::Integer, out_chs::Integer, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))

ConvDown(in_chs::Integer, out_chs::Integer, kernel = (4,4)) =
  Chain(Conv(kernel,in_chs=>out_chs,pad=(1,1),stride=(2,2);init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))

struct UNetUpBlock
  upsample
end

@functor UNetUpBlock

UNetUpBlock(in_chs::Integer, out_chs::Integer; kernel = (3, 3), p = 0.5f0) = 
    UNetUpBlock(Chain(x->leakyrelu.(x,0.2f0),
       		ConvTranspose((2, 2), in_chs=>out_chs,
			stride=(2, 2);init=_random_normal),
		BatchNormWrap(out_chs),
		Dropout(isnothing(testing_seed) ? p : 0f0)))

function (u::UNetUpBlock)(x::AbstractArray{T, 4}, bridge::AbstractArray{T, 4}) where T
  x = u.upsample(x)
  return cat(x, bridge, dims = 3)
end

"""
    Unet(channels::Int = 1, labels::Int = channels)

  Initializes a [UNet](https://arxiv.org/pdf/1505.04597.pdf) instance with the given number of `channels`, typically equal to the number of channels in the input images.
  `labels`, equal to the number of input channels by default, specifies the number of output channels.
"""
struct Unet{D}
  conv_down_blocks
  init_conv_block
  conv_blocks
  up_blocks
  out_blocks
end

@functor Unet

Unet(conv_down_blocks, init_conv_block, conv_blocks, up_blocks, out_blocks) = Unet{length(conv_down_blocks)}(conv_down_blocks, init_conv_block, conv_blocks, up_blocks, out_blocks)

function Unet(channels::Integer = 1, labels::Int = channels, depth::Integer=5)

  init_conv_block = channels >= 3 ? UNetConvBlock(channels, 64) : Chain(UNetConvBlock(channels, 3), UNetConvBlock(3, 64))

  conv_blocks = tuple([i == depth ? (x -> x) : ConvDown(2^(5+i), 2^(5+i)) for i=1:depth]...)

  conv_down_blocks = tuple([UNetConvBlock(2^(5+i), 2^(6+min(i, depth-1))) for i=1:depth]...)

  up_blocks = tuple([UNetUpBlock(2^(6+min(i+1, depth-1)), 2^(5+i); p=(i == 1 ? 0f0 : .5f0)) for i=1:depth-1]...)

  out_blocks = Chain(x -> leakyrelu.(x, 0.2f0), Conv((1, 1), 128=>labels; init=_random_normal), x -> tanh.(x))

  return Unet{depth}(conv_down_blocks, init_conv_block, conv_blocks, up_blocks, out_blocks)
end

function (u::Unet{D})(x::AbstractArray{T, 4}) where {D, T}
  x0 = u.init_conv_block(x)

  if D>0
    x0 = _unet(u, x0, Val(1))
  end

  out = u.out_blocks(x0)

  return out
end

function _unet(u::Unet{D}, x::AbstractArray{T, 4}, ::Val{d}) where {D, d, T}
  xl = u.conv_down_blocks[d](x)
  _next_level = _unet(u, xl, Val(d+1))
  xu = u.up_blocks[d](_next_level, xl)
  return u.conv_blocks[d](xu)
end

_unet(u::Unet{D}, x::AbstractArray{T, 4}, ::Val{D}) where {D, T} = x

Base.show(io::IO, u::Unet) = println(io, "UNet: no display (TBD)")
