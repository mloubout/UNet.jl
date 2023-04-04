using Test, UNet
using UNet.Flux, UNet.Flux.Zygote

@testset "Inference" begin

  for ch in (1,3)
    u = Unet(ch)
    ip = rand(Float32, 256, 256, ch, 1)

    @test size(u(ip)) == size(ip)
  end

  u = Unet(2,5)
  ip = rand(Float32, 256, 256, 2, 1)
  output = u(ip)
  @test size(output) == (256, 256, 5, 1) 
end

@testset "Variable Sizes" begin

  u = Unet(;pad_input=false)
  # test powers of 2 don't throw and return correct shape
  for s in (64, 128, 256)
    ip = rand(Float32, s, s, 1, 1)
    @test size(u(ip)) == size(ip)
  end

  broken_ip = rand(Float32, 399, 399, 1, 1)
  @test_throws DimensionMismatch size(u(broken_ip)) == size(broken_ip)
end


@testset "Padding" begin
    for D in [3, 4, 5]
        u = Unet(1, 1, D)
        x = randn(Float32, 73, 93, 1, 1)
        xu = u(x)
        @test size(xu) == size(x)
    end
end

@testset "Gradient Tests" begin
  u = Unet()
  ip = rand(Float32, 256, 256, 1,1)
  gs = gradient(Flux.params(u)) do
    sum(u(ip))
  end

  @test gs isa Zygote.Grads
end
