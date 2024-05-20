# function BatchNormWrap(out_ch::Integer)
#     Chain(x->expand_dims(x,3), BatchNorm(out_ch), x->squeeze(x))
# end

function BatchNormWrap(out_ch::Integer)
    Chain(InstanceNorm(out_ch))
end

UNetConvBlock(in_chs::Integer, out_chs::Integer; kernel = (3, 3), pad=(1,1)) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = pad;init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))

ConvDown(in_chs::Integer, out_chs::Integer; kernel = (4,4), pad=(1,1),stride=(2,2)) =
  Chain(Conv(kernel,in_chs=>out_chs,pad=pad,stride=stride;init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))

struct UNetUpBlock
  upsample
end

@functor UNetUpBlock

UNetUpBlock(in_chs::Integer, out_chs::Integer; kernel = (2, 2), p = 0.5f0) = 
    UNetUpBlock(Chain(x->leakyrelu.(x,0.2f0),
       		ConvTranspose(kernel, in_chs=>out_chs,
			stride=kernel;init=_random_normal),
		BatchNormWrap(out_chs),
		Dropout(isnothing(testing_seed) ? p : 0f0)))

function (u::UNetUpBlock)(x::AbstractArray{T, N}, bridge::AbstractArray{T, N}) where {T, N}
  x = u.upsample(x)
  return cat(x, bridge, dims = N-1)
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

function Unet(channels::Integer = 1, labels::Int = channels, depth::Integer=5; dims=3)

  pad = Tuple(1 for i=1:dims)
  stride = Tuple(2 for i=1:dims)
  kernel_1 = Tuple(1 for i=1:dims)
  kernel_2 = Tuple(2 for i=1:dims)
  kernel_3 = Tuple(3 for i=1:dims)
  kernel_4 = Tuple(4 for i=1:dims)

  init_conv_block = channels >= 3 ? UNetConvBlock(channels, 64;kernel=kernel_3,pad=pad) : Chain(UNetConvBlock(channels, 4;kernel=kernel_3,pad=pad), UNetConvBlock(4, 64;kernel=kernel_3,pad=pad))

  conv_down_blocks = tuple([i == depth ? x -> x : ConvDown(2^(5+i), 2^(5+i);kernel=kernel_4,pad=pad,stride=stride) for i=1:depth]...)

  conv_blocks = tuple([UNetConvBlock(2^(5+i), 2^(6+min(i, depth-1));kernel=kernel_3,pad=pad) for i=1:depth]...)

  up_blocks = tuple([UNetUpBlock(2^(6+min(i+1, depth-1)), 2^(5+i);kernel=kernel_2, p=(i == 1 ? 0f0 : .5f0)) for i=depth-1:-1:1]...)

  out_blocks = Chain(x -> leakyrelu.(x, 0.2f0), Conv(kernel_1, 128=>labels; init=_random_normal), x -> x)

  return Unet{depth}(conv_down_blocks, init_conv_block, conv_blocks, up_blocks, out_blocks)
end

function (u::Unet{D})(x::AbstractArray{T, N}) where {D, T,N}
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

function Base.show(io::IO, u::Unet)
  println(io, "UNet:")

  for l in u.conv_down_blocks
    println(io, "  ConvDown($(size(l[1].weight)[end-1]), $(size(l[1].weight)[end]))")
  end

  println(io, "\n")
  for l in u.conv_blocks
    println(io, "  UNetConvBlock($(size(l[1].weight)[end-1]), $(size(l[1].weight)[end]))")
  end

  println(io, "\n")
  for l in u.up_blocks
    l isa UNetUpBlock || continue
    println(io, "  UNetUpBlock($(size(l.upsample[2].weight)[end]), $(size(l.upsample[2].weight)[end-1]))")
  end
end
