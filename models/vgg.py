import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

warnings.filterwarnings(
    "ignore", message="Setting attributes on ParameterList is not supported."
)

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

class VGG1D(nn.Module):
    def __init__(
        self,
        vgg_name,
        num_ch=3,
        num_classes=10,
        bias=True,
        batch_norm=True,
        pooling="max",
        pooling_size=2,
        param_list=False,
        width_factor=1,
        stride=2,
    ):
        super(VGG1D, self).__init__()
        if pooling == True:
            pooling = "max"
        self.features = self._make_layers(
            cfg[vgg_name],
            ch=num_ch,
            bias=bias,
            bn=batch_norm,
            pooling=pooling,
            ps=pooling_size,
            param_list=param_list,
            width_factor=width_factor,
            stride=stride
        )
        stride_factor = 729 if stride == 1 else 1
        self.classifier = nn.Linear(int(512 * width_factor) * stride_factor, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, ch, bias, bn, pooling, ps, param_list, width_factor, stride):
        layers = []
        in_channels = ch
        # Calculate padding for input size 36
        padding_36 = max(0, ((36 - 1) * stride - 32 + ps - 1) // 2)
        if ch == 1:
            layers.append(nn.ZeroPad1d(padding_36))
        if param_list:
            convLayer = Conv1dList
        else:
            convLayer = nn.Conv1d
        for x in cfg:
            if x == "M":
                if pooling == "max":
                    layers += [
                        nn.MaxPool1d(
                            kernel_size=ps, stride=stride, padding=ps // 2 + ps % 2 - 1
                        )
                    ]
                elif pooling == "avg":
                    layers += [
                        nn.AvgPool1d(
                            kernel_size=ps, stride=stride, padding=ps // 2 + ps % 2 - 1
                        )
                    ]
                else:
                    layers += [SubSampling1D(kernel_size=ps, stride=stride)]
            else:
                x = int(x * width_factor)
                if bn:
                    layers += [
                        convLayer(in_channels, x, kernel_size=3, padding=1, bias=bias),
                        nn.BatchNorm1d(x),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        convLayer(in_channels, x, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    ]
                in_channels = x
        return nn.Sequential(*layers)


class Conv1dList(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="constant",
    ):
        super().__init__()

        weight = torch.empty(out_channels, in_channels, kernel_size)

        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if bias is not None:
            bias = nn.Parameter(
                torch.empty(
                    out_channels,
                )
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        n = max(1, 128 * 256 // (in_channels * kernel_size)) # * kernel_size
        weight = nn.ParameterList(
            [nn.Parameter(weight[j : j + n]) for j in range(0, len(weight), n)]
        )

        setattr(self, "weight", weight)
        setattr(self, "bias", bias)

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride

    def forward(self, x):

        weight = self.weight
        if isinstance(weight, nn.ParameterList):
            weight = torch.cat(list(self.weight))

        return F.conv1d(x, weight, self.bias, self.stride, self.padding)


class SubSampling1D(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(SubSampling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.unfold(2, self.kernel_size, self.stride)[..., 0]


