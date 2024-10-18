import functools
import torch


class Resnet2D(torch.nn.Module):
    """Resnet that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=None,
        ngf=64,
        norm_layer=torch.nn.InstanceNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        activation=torch.nn.ReLU,
        n_downsampling=2,
    ):
        """Construct a Resnet
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images (default is ngf)
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zeros | valid
            activation          -- non-linearity layer to apply (default is ReLU)
            n_downsampling      -- number of times to downsample data before ResBlocks
        """
        assert n_blocks >= 0
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d

        if output_nc is None:
            output_nc = ngf

        p = 0
        updown_p = 1
        padder = []
        if padding_type.lower() == "reflect" or padding_type.lower() == "same":
            padder = [torch.nn.ReflectionPad2d(3)]
        elif padding_type.lower() == "replicate":
            padder = [torch.nn.ReplicationPad2d(3)]
        elif padding_type.lower() == "zeros":
            p = 3
        elif padding_type.lower() == "valid":
            p = "valid"
            updown_p = 0

        model = []
        model += padder.copy()
        model += [
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=p, bias=use_bias),
            norm_layer(ngf),
            activation(),
        ]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [
                torch.nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=updown_p,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                activation(),
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock2D(
                    ngf * mult,
                    padding_type=padding_type.lower(),
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    activation=activation,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                torch.nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=updown_p,
                    output_padding=updown_p,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation(),
            ]
        model += padder.copy()
        model += [torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=p)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock2D(torch.nn.Module):
    """Define a Resnet block"""

    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        activation=torch.nn.ReLU,
    ):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, activation
        )
        self.padding_type = padding_type

    def build_conv_block(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        activation=torch.nn.ReLU,
    ):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zeros | valid
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
            activation          -- non-linearity layer to apply (default is ReLU)
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        p = 0
        padder = []
        if padding_type == "reflect" or padding_type.lower() == "same":
            padder = [torch.nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            padder = [torch.nn.ReplicationPad2d(1)]
        elif padding_type == "zeros":
            p = 1
        elif padding_type == "valid":
            p = "valid"
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block = []
        conv_block += padder.copy()

        conv_block += [
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            activation(),
        ]
        if use_dropout:
            conv_block += [torch.nn.Dropout(0.2)]

        conv_block += padder.copy()
        conv_block += [
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return torch.nn.Sequential(*conv_block)

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[:-2] + shape

        offset = tuple(
            torch.div((a - b), 2, rounding_mode="trunc")
            for a, b in zip(x.size(), x_target_size)
        )

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):
        """Forward function (with skip connections)"""
        if self.padding_type == "valid":  # crop for valid networks
            res = self.conv_block(x)
            out = self.crop(x, res.size()[-2:]) + res
        else:
            out = x + self.conv_block(x)  # add skip connections
        return out


class Resnet3D(torch.nn.Module):
    """Resnet that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=None,
        ngf=64,
        norm_layer=torch.nn.InstanceNorm3d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        activation=torch.nn.ReLU,
        n_downsampling=2,
    ):
        """Construct a Resnet
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zeros | valid
            activation          -- non-linearity layer to apply (default is ReLU)
            n_downsampling      -- number of times to downsample data before ResBlocks
        """
        assert n_blocks >= 0
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm3d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm3d

        if output_nc is None:
            output_nc = ngf

        p = 0
        updown_p = 1
        padder = []
        if padding_type.lower() == "reflect" or padding_type.lower() == "same":
            padder = [torch.nn.ReflectionPad3d(3)]
        elif padding_type.lower() == "replicate":
            padder = [torch.nn.ReplicationPad3d(3)]
        elif padding_type.lower() == "zeros":
            p = 3
        elif padding_type.lower() == "valid":
            p = "valid"
            updown_p = 0

        model = []
        model += padder.copy()
        model += [
            torch.nn.Conv3d(input_nc, ngf, kernel_size=7, padding=p, bias=use_bias),
            norm_layer(ngf),
            activation(),
        ]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [
                torch.nn.Conv3d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=updown_p,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                activation(),
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock3D(
                    ngf * mult,
                    padding_type=padding_type.lower(),
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    activation=activation,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                torch.nn.ConvTranspose3d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=updown_p,
                    output_padding=updown_p,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation(),
            ]
        model += padder.copy()
        model += [torch.nn.Conv3d(ngf, output_nc, kernel_size=7, padding=p)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock3D(torch.nn.Module):
    """Define a Resnet block"""

    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        activation=torch.nn.ReLU,
    ):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, activation
        )
        self.padding_type = padding_type

    def build_conv_block(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        activation=torch.nn.ReLU,
    ):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zeros | valid
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
            activation          -- non-linearity layer to apply (default is ReLU)
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        p = 0
        padder = []
        if padding_type == "reflect" or padding_type.lower() == "same":
            padder = [torch.nn.ReflectionPad3d(1)]
        elif padding_type == "replicate":
            padder = [torch.nn.ReplicationPad3d(1)]
        elif padding_type == "zeros":
            p = 1
        elif padding_type == "valid":
            p = "valid"
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block = []
        conv_block += padder.copy()

        conv_block += [
            torch.nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            activation(),
        ]
        if use_dropout:
            conv_block += [torch.nn.Dropout(0.2)]

        conv_block += padder.copy()
        conv_block += [
            torch.nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return torch.nn.Sequential(*conv_block)

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[:-3] + shape

        offset = tuple(
            torch.div((a - b), 2, rounding_mode="trunc")
            for a, b in zip(x.size(), x_target_size)
        )

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):
        """Forward function (with skip connections)"""
        if self.padding_type == "valid":  # crop for valid networks
            res = self.conv_block(x)
            out = self.crop(x, res.size()[-3:]) + res
        else:
            out = x + self.conv_block(x)  # add skip connections
        return out


class ResNet(Resnet2D, Resnet3D):
    """Resnet that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, ndims, **kwargs):
        """Construct a Resnet
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images (default is ngf)
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zeros | valid
            activation          -- non-linearity layer to apply (default is ReLU)
            n_downsampling      -- number of times to downsample data before ResBlocks
        """
        if ndims == 2:
            Resnet2D.__init__(self, **kwargs)
        elif ndims == 3:
            Resnet3D.__init__(self, **kwargs)
        else:
            raise ValueError(
                ndims,
                "Only 2D or 3D currently implemented. Feel free to contribute more!",
            )
