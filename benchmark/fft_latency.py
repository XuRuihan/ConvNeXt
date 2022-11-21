import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark


class SpatialOversizeConv2d(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, interpolate=False):
        super().__init__()
        if interpolate is not True:
            assert kernel_size % 2 == 1
            padding = kernel_size // 2
        else:
            padding = 0

        self.conv_h = nn.Conv2d(
            dim, dim, (kernel_size, 1), padding=(padding, 0), groups=dim, bias=bias
        )
        self.conv_w = nn.Conv2d(
            dim, dim, (1, kernel_size), padding=(0, padding), groups=dim, bias=bias
        )

        self.dim = dim
        self.kernel_size = kernel_size
        self.interpolate = interpolate
        self.padding = padding

    def get_instance_kernel(self, instance_kernel_size):
        h_weight = F.interpolate(
            self.conv_h.weight,
            [instance_kernel_size[0], 1],
            mode="bilinear",
            align_corners=True,
        )
        w_weight = F.interpolate(
            self.conv_w.weight,
            [1, instance_kernel_size[1]],
            mode="bilinear",
            align_corners=True,
        )
        return h_weight, w_weight

    def forward(self, x):
        if self.interpolate:
            H, W = x.shape[-2:]
            instance_kernel_size = 2 * H - 1, 2 * W - 1
            h_weight, w_weight = self.get_instance_kernel(instance_kernel_size)
            padding = H - 1, W - 1

            x = F.conv2d(
                x, h_weight, self.conv_h.bias, padding=(padding[0], 0), groups=self.dim
            )
            x = F.conv2d(
                x, w_weight, self.conv_w.bias, padding=(0, padding[1]), groups=self.dim
            )
        else:
            x = self.conv_h(x)
            x = self.conv_w(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, kernel_size={self.kernel_size}"


class FourierOversizeConv2d(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, interpolate=False):
        super().__init__()
        if interpolate is not True:
            assert kernel_size % 2 == 1
            padding = kernel_size // 2
        else:
            padding = 0

        self.conv_h = nn.Conv2d(
            dim, dim, (kernel_size, 1), padding=(padding, 0), groups=dim, bias=bias
        )
        self.conv_w = nn.Conv2d(
            dim, dim, (1, kernel_size), padding=(0, padding), groups=dim, bias=bias
        )

        self.dim = dim
        self.kernel_size = kernel_size
        self.interpolate = interpolate
        self.padding = padding

    def get_instance_kernel(self, instance_kernel_size):
        h_weight = F.interpolate(
            self.conv_h.weight,
            [instance_kernel_size[0], 1],
            mode="bilinear",
            align_corners=True,
        )
        w_weight = F.interpolate(
            self.conv_w.weight,
            [1, instance_kernel_size[1]],
            mode="bilinear",
            align_corners=True,
        )
        return h_weight, w_weight

    def forward(self, x):
        if self.interpolate:
            H, W = x.shape[-2:]
            instance_kernel_size = 2 * H - 1, 2 * W - 1
            h_weight, w_weight = self.get_instance_kernel(instance_kernel_size)
            padding = H - 1, W - 1

            x = F.conv2d(
                x, h_weight, self.conv_h.bias, padding=(padding[0], 0), groups=self.dim
            )
            x = F.conv2d(
                x, w_weight, self.conv_w.bias, padding=(0, padding[1]), groups=self.dim
            )
        else:
            H, W = x.shape[-2:]
            x = F.pad(x, [self.padding, 0, self.padding, 0])
            x_fft = torch.fft.fft2(x)
            h_weight = torch.fft.fft2(self.conv_h.weight).squeeze(1)
            w_weight = torch.fft.fft2(self.conv_w.weight).squeeze(1)
            x_fft = x_fft * torch.conj(h_weight) * torch.conj(w_weight)
            x = torch.fft.ifft2(x).real
            x = F.pad(x, [0, -self.padding, 0, -self.padding])
        return x

    def extra_repr(self):
        return f"dim={self.dim}, kernel_size={self.kernel_size}"


kernel_size = 255
spatial_net = SpatialOversizeConv2d(256, kernel_size, bias=False).eval().cuda()
fourier_net = FourierOversizeConv2d(256, kernel_size, bias=False).eval().cuda()
x = torch.randn(10, 256, 128, 128).cuda()


def spatial_conv(x):
    return spatial_net(x)


def fourier_conv(x):
    return fourier_net(x)


t0 = benchmark.Timer(
    stmt="spatial_conv(x)", setup="from __main__ import spatial_conv", globals={"x": x}
)

t1 = benchmark.Timer(
    stmt="fourier_conv(x)", setup="from __main__ import fourier_conv", globals={"x": x}
)

print(t0.timeit(100))
print(t1.timeit(100))
