import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark


# ParC V1
def ParC(x, weight, bias=None):
    B, C, H, W = x.shape
    x_cat = torch.cat((x, x[:, :, :-1, :]), dim=-2)
    x = F.conv2d(x_cat, weight, bias, groups=C)
    return x


def Fast_ParC(x, weight, bias=None):
    B, C, H, W = x.shape
    x = torch.fft.rfft(x, dim=-2)
    weight = torch.fft.rfft(weight, dim=-2)
    x = x * torch.conj(weight.squeeze(1))
    x = torch.fft.irfft(x, dim=-2)
    # if bias is not None:
    #     x = x + bias.view(1, C, 1, 1)
    return x


# ParC V2
def ParC_V2(x, weight, bias=None):
    B, C, H, W = x.shape
    padding = H - 1
    x = F.conv2d(x, weight, bias, padding=(padding, 0), groups=C)
    return x


def Fast_ParC_V2(x, weight, bias=None):
    B, C, H, W = x.shape
    padding = H - 1
    x = F.pad(x, [0, 0, padding, 0])
    x = torch.fft.fft(x, dim=-2)
    weight = torch.fft.fft(weight, dim=-2)
    x = x * torch.conj(weight.squeeze(1))
    x = torch.fft.ifft(x, n=2 * H - 1, dim=-2).real
    x = F.pad(x, [0, 0, 0, -padding])
    if bias is not None:
        x = x + bias.view(1, C, 1, 1)
    return x


def parcnet():
    dim = 256
    kernel_size = 256
    image_size = 256
    x = torch.rand(10, dim, image_size, image_size).cuda(1)
    weight = torch.rand(dim, 1, kernel_size, 1).cuda(1)
    bias = torch.rand(dim).cuda(1)
    bias = None

    # FFT cannot reach a very high accuracy such as 1e-8.
    assert ParC(x, weight, bias).allclose(Fast_ParC(x, weight, bias), atol=1e-4)

    t0 = benchmark.Timer(
        stmt="ParC(x, weight, bias)",
        setup="from __main__ import ParC",
        globals={"x": x, "weight": weight, "bias": bias},
    )

    t1 = benchmark.Timer(
        stmt="Fast_ParC(x, weight, bias)",
        setup="from __main__ import Fast_ParC",
        globals={"x": x, "weight": weight, "bias": bias},
    )

    print("==================== ParC inference latent ====================")
    for i in range(4):
        print(t0.timeit(100))
        print(t1.timeit(100))


def parcnet_v2():
    dim = 256
    kernel_size = 255
    image_size = 128
    x = torch.rand(10, dim, image_size, image_size).cuda(1)
    weight = torch.rand(dim, 1, kernel_size, 1).cuda(1)
    bias = torch.rand(dim).cuda(1)

    # FFT cannot reach a very high accuracy such as 1e-8.
    assert ParC_V2(x, weight, bias).allclose(Fast_ParC_V2(x, weight, bias), atol=1e-4)

    t0 = benchmark.Timer(
        stmt="ParC_V2(x, weight, bias)",
        setup="from __main__ import ParC_V2",
        globals={"x": x, "weight": weight, "bias": bias},
    )

    t1 = benchmark.Timer(
        stmt="Fast_ParC_V2(x, weight, bias)",
        setup="from __main__ import Fast_ParC_V2",
        globals={"x": x, "weight": weight, "bias": bias},
    )

    print("==================== ParC V2 inference latent ====================")
    for i in range(4):
        print(t0.timeit(100))
        print(t1.timeit(100))


if __name__ == "__main__":
    parcnet()
    # parcnet_v2()
