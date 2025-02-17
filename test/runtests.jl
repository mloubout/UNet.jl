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

  u = Unet()
  # test powers of 2 don't throw and return correct shape
  for s in (64, 128, 256)
    ip = rand(Float32, s, s, 1, 1)
    @test size(u(ip)) == size(ip)
  end

  broken_ip = rand(Float32, 299, 299, 1, 1)
  @test_throws DimensionMismatch size(u(broken_ip)) == size(broken_ip)
end

@testset "Legacy Tests" begin
  # Prevent random initialization to compare outputs
  UNet.test_mode()
  for chi in (1, 3)
    for cho in (1, 3)
      u = Unet(chi, cho)
      ul = LegacyUnet(chi, cho)
      ip = rand(Float32, 256, 256, chi, 1)
      op = u(ip)
      opl = ul(ip)
      @test op ≈ opl
    end
  end
  UNet.run_mode()
end


@testset "Gradient Tests" begin
  u = Unet()
  ip = rand(Float32, 256, 256, 1,1)
  gs = gradient(Flux.params(u)) do
    sum(u(ip))
  end

  @test gs isa Zygote.Grads
end
