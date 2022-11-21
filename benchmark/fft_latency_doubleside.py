import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark


# ParC V1
def ParC(x, h_weight, w_weight, bias=None):
    B, C, H, W = x.shape
    x_cat = torch.cat((x, x[:, :, :-1, :]), dim=-2)
    x = F.conv2d(x_cat, h_weight, groups=C)
    x_cat = torch.cat((x, x[:, :, :, :-1]), dim=-1)
    x = F.conv2d(x_cat, w_weight, bias, groups=C)
    return x


def Fast_ParC(x, h_weight, w_weight, bias=None):
    B, C, H, W = x.shape
    x = torch.fft.rfft2(x)
    h_weight = torch.fft.rfft2(h_weight)
    w_weight = torch.fft.rfft2(w_weight)
    x = x * torch.conj(h_weight.squeeze(1)) * torch.conj(w_weight.squeeze(1))
    x = torch.fft.irfft2(x, s=(H, W))
    if bias is not None:
        x = x + bias.view(1, C, 1, 1)
    return x


# ParC V2
def ParC_V2(x, h_weight, w_weight, bias=None):
    B, C, H, W = x.shape
    padding = H - 1
    x = F.conv2d(x, h_weight, padding=(padding, 0), groups=C)
    x = F.conv2d(x, w_weight, bias, padding=(0, padding), groups=C)
    return x


def Fast_ParC_V2(x, h_weight, w_weight, bias=None):
    B, C, H, W = x.shape
    padding = H - 1
    x = F.pad(x, [W - 1, 0, H - 1, 0])
    x = torch.fft.rfft2(x)
    h_weight = torch.fft.rfft2(h_weight)
    w_weight = torch.fft.rfft2(w_weight)
    x = x * torch.conj(h_weight.squeeze(1)) * torch.conj(w_weight.squeeze(1))
    x = torch.fft.irfft2(x, s=(2 * H - 1, 2 * W - 1))
    x = x[:, :, :H, :W]
    if bias is not None:
        x = x + bias.view(1, C, 1, 1)
    return x


def parcnet():
    dim = 1
    kernel_size = 64
    image_size = 64
    device = "cuda:1"
    x = torch.rand(10, dim, image_size, image_size).to(device)
    h_weight = torch.rand(dim, 1, kernel_size, 1).to(device)
    w_weight = torch.rand(dim, 1, 1, kernel_size).to(device)
    bias = torch.rand(dim).to(device)
    # bias = None

    # FFT cannot reach a very high accuracy such as 1e-8.
    assert ParC(x, h_weight, w_weight, bias).allclose(
        Fast_ParC(x, h_weight, w_weight, bias), atol=1e-8
    )

    t0 = benchmark.Timer(
        stmt="ParC(x, h_weight, w_weight, bias)",
        setup="from __main__ import ParC",
        globals={"x": x, "h_weight": h_weight, "w_weight": w_weight, "bias": bias},
    )

    t1 = benchmark.Timer(
        stmt="Fast_ParC(x, h_weight, w_weight, bias)",
        setup="from __main__ import Fast_ParC",
        globals={"x": x, "h_weight": h_weight, "w_weight": w_weight, "bias": bias},
    )

    print("==================== ParC inference latent ====================")
    print(t0.timeit(100))
    print(t1.timeit(100))


def parcnet_v2():
    dim = 1
    kernel_size = 255
    image_size = 128
    device = "cuda:1"
    x = torch.rand(10, dim, image_size, image_size).to(device)
    h_weight = torch.rand(dim, 1, kernel_size, 1).to(device)
    w_weight = torch.rand(dim, 1, 1, kernel_size).to(device)
    bias = torch.rand(dim).to(device)
    # bias = None

    # FFT cannot reach a very high accuracy such as 1e-8.
    assert ParC_V2(x, h_weight, w_weight, bias).allclose(
        Fast_ParC_V2(x, h_weight, w_weight, bias), atol=1e-8
    )

    t0 = benchmark.Timer(
        stmt="ParC_V2(x, h_weight, w_weight, bias)",
        setup="from __main__ import ParC_V2",
        globals={"x": x, "h_weight": h_weight, "w_weight": w_weight, "bias": bias},
    )

    t1 = benchmark.Timer(
        stmt="Fast_ParC_V2(x, h_weight, w_weight, bias)",
        setup="from __main__ import Fast_ParC_V2",
        globals={"x": x, "h_weight": h_weight, "w_weight": w_weight, "bias": bias},
    )

    print("==================== ParC V2 inference latent ====================")
    print(t0.timeit(100))
    print(t1.timeit(100))


if __name__ == "__main__":
    parcnet()
    parcnet_v2()
