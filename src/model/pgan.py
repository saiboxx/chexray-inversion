"""Classes and functionsare obtained  from https://github.com/BradSegal/CXR_PGGAN."""
from math import prod, sqrt
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def he_initializer(module: nn.Module) -> float:
    """
    Return He's Initialization Constant for Conv2D or linear modules.

    It is inversely proportional to the root of the product of the neurons/weights for a
    given module. Scales the gradient relative to the number of weights
    to remove the correlation between the number of connections and the gradient.
    Formulation only valid for convolutional & linear layers due to weight arrangement
    https://arxiv.org/abs/1502.01852
    """
    assert isinstance(
        module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    ), 'Formulation only valid for Conv2d & linear layers'
    weight_shape = module.weight.size()
    # Removes the out_channel weights and multiplies the rest together
    fan_in = prod(weight_shape[1:])
    he_const = sqrt(2.0 / fan_in)

    return he_const


class PixelNormalizationLayer(nn.Module):
    """
    Pixel normalization layer module.

    Normalizes a minibatch of images by dividing each pixel by the average squared pixel
    across all channels Norm = Root(Pixel / Sum(Pixel**2)/(Num Channels))
    """

    def forward(self, x: Tensor, epsilon=1e-8) -> Tensor:
        """Pass a Tensor through the layer."""
        # Epsilon → Small value for numerical stability when dividing
        # rsqrt → Reciprocal Square Root
        return x * (x.pow(2).mean(dim=1, keepdim=True) + epsilon).rsqrt()


class EqualizedLayer(nn.Module):
    """
    Equalized layer module.

    Wrapper layer that enables a linear or convolutional layer to execute He
    Initialization at runtime as well as set initial biases of a module to 0.
    The initialization is performed during the forward pass of the network to enable
    adaptive gradient descent methods (eg. Adam) to better compensate for the
    equalization of learning rates. Equalization first sets all weights to random
    numbers between -1 & 1 / N(0, 1), and then multiplies by the He constant at runtime.
    """

    def __init__(
        self,
        module: nn.Module,
        equalize: bool = True,
        bias_init: bool = True,
        lrmult: float = 1.0,
    ) -> None:
        """
        Initialize Equalized layer.

        :param module: Torch module to be equalized based on the number of connections
        :param equalize: Flag to disable He Initialization
        :param bias_init: Flag to disable initializing bias values to 0
        :param lrmult: Custom layer-specific learning rate multiplier
        """
        super().__init__()

        self.module = module
        self.equalize = equalize
        self.init_bias = bias_init

        if self.equalize:
            self.module.weight.data.normal_(0, 1)
            # Scale weights by a layer specific learning rate multiplier.
            # Divides by multiplier as the He Value is the
            # reciprocal of multiple of the output weights
            self.module.weight.data /= lrmult
            self.he_val = he_initializer(self.module)
        if self.init_bias:
            self.module.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward a tensor through the network."""
        x = self.module(x)
        # Scale by the He Constant
        if self.equalize:
            x *= self.he_val
        return x


class EqualizedConv2D(EqualizedLayer):
    """
    Equalized conv layer module.

    Modified 2D convolution that is able to employ He Initialization at runtime as well
    as to initialize biases to 0.
    """

    def __init__(
        self,
        prev_channels: int,
        channels: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        transpose: bool = False,
        **kwargs
    ) -> None:
        """Initialize Equalized layer."""
        if not transpose:
            conv = nn.Conv2d(
                in_channels=prev_channels,
                out_channels=channels,
                kernel_size=(kernel, kernel),
                stride=(stride, stride),
                padding=padding,
                bias=bias,
            )
        else:
            conv = nn.ConvTranspose2d(
                in_channels=prev_channels,
                out_channels=channels,
                kernel_size=(kernel, kernel),
                stride=(stride, stride),
                padding=(padding, padding),
                bias=bias,
            )

        EqualizedLayer.__init__(self, conv, **kwargs)


class EqualizedLinear(EqualizedLayer):
    """
    Equalized conv layer module.

    Modified Fully Connected Layer to employ He Initialization at runtime and
    initialize biases to 0.
    """

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        """Initialize Equalized layer."""
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        linear = nn.Linear(in_features, out_features, bias=bias)
        EqualizedLayer.__init__(self, linear, **kwargs)


def make_conv_block(
    prev_channel: int,
    channels: int,
    kernel: int,
    stride: int,
    padding: int,
    bias: bool = True,
    equalize: bool = True,
    leakiness: float = 0.2,
    normalize: bool = True,
    activation: bool = True,
    transpose: bool = False,
) -> nn.ModuleList:
    """Create a convolutional block."""
    block = nn.ModuleList()
    if equalize:
        block.append(
            EqualizedConv2D(
                prev_channels=prev_channel,
                channels=channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
                bias=bias,
                transpose=transpose,
            )
        )
    else:
        block.append(
            nn.Conv2d(
                in_channels=prev_channel,
                out_channels=channels,
                kernel_size=(kernel, kernel),
                stride=(stride, stride),
                padding=padding,
                bias=bias,
            )
        )
    if activation:
        block.append(nn.LeakyReLU(negative_slope=leakiness, inplace=True))

    if normalize:
        block.append(PixelNormalizationLayer())

    return block


class ScaleBlock(nn.Module):
    """
    Scale Block module.

    Standard convolutional block that combines two identical convolutions and an
    interpolation operation.
    If the block upscales an image, the upscaling is done prior to the convolutions
    If the block downscales, the upscaling is done after the convolutions
    """

    def __init__(
        self,
        dims: Tensor,
        prev_channel: int,
        channels: int,
        scale: float = 1,
        equalize: bool = True,
        normalize: bool = True,
        leakiness: float = 0.2,
        kernel: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        mode: str = 'bilinear',
    ) -> None:
        """Initialize ScaleBlock module."""
        super().__init__()

        assert scale in [
            0.5,
            1,
            2,
        ], 'Scale can only half, double or maintain spatial resolution'
        self.scale = scale
        self.equalize = equalize

        assert mode in [
            'nearest',
            'bilinear',
        ], 'Only configured for "nearest" & "bilinear", but {} was selected'.format(
            mode
        )
        self.mode = mode

        if padding is None:
            padding = calc_padding(dims, kernel, stride)

        self.convolv = nn.Sequential(
            *make_conv_block(
                prev_channel=prev_channel,
                channels=channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
                bias=bias,
                equalize=equalize,
                leakiness=leakiness,
                normalize=normalize,
            ),
            *make_conv_block(
                prev_channel=channels,
                channels=channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
                bias=bias,
                equalize=equalize,
                leakiness=leakiness,
                normalize=normalize,
            )
        )

    def forward(self, feat_map: Tensor) -> Tensor:
        """Forward a tensor through the network."""
        if self.scale > 1:
            feat_map = F.interpolate(
                input=feat_map, scale_factor=self.scale, mode=self.mode
            )

        feat_map = self.convolv(feat_map)

        if self.scale < 1:
            feat_map = F.avg_pool2d(input=feat_map, kernel_size=(2, 2))

        return feat_map


def calc_padding(dims: Tensor, kernel: int, stride: int) -> int:
    """
    Compute fitting padding size.

    For constant output: W = (W - F + 2P)/S + 1
    W → Dimensions, F → Kernel, P → Padding, S → Stride
    (S(W - 1) - W + F)/2
    """
    padding = (stride * (dims - 1) - dims + kernel) / 2
    assert padding.is_integer(), (
        'A non-integer result indicates an invalid pairing of '
        'dimensions and stride values'
    )
    return int(padding)


class ProGenerator(nn.Module):
    """Progressive growing generator."""

    def __init__(
        self,
        z_dim: int = 512,
        channel_depth: int = 512,
        init_bias: bool = True,
        norm_layers: bool = True,
        out_channels: int = 3,
        equalize_layers: bool = True,
        leakiness: float = 0.2,
        mode: str = 'nearest',
    ) -> None:
        """Initialize ProGenerator module."""
        super().__init__()
        # Dimensions of latent space vector
        self.z_dim = z_dim
        # Initial Number of channels to produce from latent space
        self.channel_depth = [channel_depth]
        # Model begins by producing 4x4 images, incremented when alpha reaches 1
        # Current number of completed blocks, incremented when alpha reaches 1.
        self.register_buffer('current_size', torch.tensor(4))
        self.register_buffer('current_depth', torch.tensor(0))
        # Mixing co-efficient for use when upscaling the network
        self.alpha: Tensor
        self.register_buffer('alpha', torch.tensor(0))

        # Initialize bias to 0
        self.init_bias = init_bias
        # Whether to apply minibatch normalization layer
        self.norm_layers = norm_layers
        # The final number of color channels used in the generated image
        self.out_channels = out_channels
        # Whether to use the He Constant to equalize layer outputs at runtime
        self.equalize = equalize_layers
        # The co-efficient of the negative slope of the Leaky ReLU activation
        self.leakiness = leakiness
        # Interpolation mode for upscaling, Paper utilizes nearest neighbour mode.
        self.mode = mode

        # Define Layer Architectures
        self.latent_linear = nn.Sequential(
            EqualizedLinear(
                in_features=z_dim,
                out_features=16 * channel_depth,
                equalize=equalize_layers,
                bias_init=init_bias,
            ),
            # Initial latent space processing
            nn.LeakyReLU(negative_slope=leakiness, inplace=True),
        )

        self.init_conv = nn.Sequential(
            # Initial convolution on latent space after initial linear processing
            *make_conv_block(
                prev_channel=channel_depth,
                channels=channel_depth,
                # Convolutions maintain number of channels
                kernel=3,
                stride=1,
                padding=calc_padding(dims=4, kernel=3, stride=1),
                bias=True,
                equalize=equalize_layers,
                leakiness=leakiness,
                normalize=norm_layers,
            )
        )

        # Stores list of scaling blocks to double spatial resolutions
        self.ScaleBlocks = nn.ModuleList()
        # Stores the feature map to RGB convolutions.
        # A new one is needed for each network expansion
        self.toRGB = nn.ModuleList()
        # The RGB layers are stored to enable extracting
        # smaller intermediary images from scaling blocks
        self.toRGB.append(
            *make_conv_block(
                prev_channel=channel_depth,
                channels=out_channels,
                kernel=1,
                stride=1,
                padding=0,
                bias=True,
                equalize=equalize_layers,
                activation=False,
                normalize=False,
            )
        )

        # The final convolution acts as an activation function
        if self.norm_layers:
            self.norm = PixelNormalizationLayer()

    def forward(self, x: Tensor) -> Tensor:
        """Forward a tensor through the network."""
        if self.norm_layers:
            x = self.norm(x)

        # Multiple of all dimensions except batch dimension = Total Feature Number
        features = prod(x.size()[1:])
        x = x.view(-1, features)

        batch_size = x.size()[0]
        # Initial Latent Processing & Formatting
        x = self.latent_linear(x)
        x = x.view(batch_size, -1, 4, 4)

        # Perform initial 3x3 convolution without upscaling
        x = self.init_conv(x)

        # Apply mixing for when the network begins to expand.
        if self.alpha > 0 and self.current_depth == 1:
            expansion = self.toRGB[-2](x)
            expansion = F.interpolate(input=expansion, scale_factor=2, mode=self.mode)

        for scale_num, scale_block in enumerate(self.ScaleBlocks, 1):
            # Start at 1 due to the first image dimension not requiring scaling
            # Process the input through the expansion block of upscale, conv, conv
            x = scale_block(x)

            if self.alpha > 0 and (scale_num == self.current_depth - 1):
                expansion = self.toRGB[-2](x)
                expansion = F.interpolate(
                    input=expansion, scale_factor=2, mode=self.mode
                )

        # Final layer to RGB
        x = self.toRGB[-1](x)

        if self.alpha > 0:
            # Mix the inputs at the final scale
            x = self.alpha * expansion + (1.0 - self.alpha) * x

        return x

    def increment_depth(self, new_depth: int):
        """Add scaling block to the model and double spatial resolution."""
        device = next(self.parameters()).device
        self.current_depth += 1
        self.current_size *= 2

        prev_depth = self.channel_depth[-1]
        self.channel_depth.append(new_depth)
        # Adds scaling block.
        # Padding is calculated from the spatial dimensions and filter properties.
        size = self.current_size.cpu().numpy()
        self.ScaleBlocks.append(
            ScaleBlock(
                dims=size,
                prev_channel=prev_depth,
                channels=new_depth,
                scale=2,
                equalize=self.equalize,
                normalize=self.norm_layers,
                leakiness=self.leakiness,
                kernel=3,
                stride=1,
                padding=None,
                bias=True,
                mode=self.mode,
            ).to(device)
        )

        self.toRGB.append(
            *make_conv_block(
                prev_channel=new_depth,
                channels=self.out_channels,
                kernel=1,
                stride=1,
                padding=0,
                bias=True,
                equalize=self.equalize,
                activation=False,
                normalize=False,
            ).to(device)
        )

    def set_alpha(self, new_alpha: float) -> None:
        """
        Set alpha.

        Sets the mixing factor used when upscaling the network. Alters the functioning
        of the forward function to include the second last layer and interpolate
        between it and the final output of the added scaling block.
        """
        if new_alpha < 0 or new_alpha > 1:
            raise ValueError('Alpha must be in the range [0,1]')
        self.alpha = new_alpha

    def load(self, checkpoint: dict) -> None:
        """
        Load checkpoint.

        Automatically scales the network to the required size and loads the weights
        :param checkpoint: Saved network state
        :return:
        """
        for depth in checkpoint['settings']['channel_depth'][1:]:
            self.increment_depth(depth)
        self.load_state_dict(checkpoint['state_dict'])
        print('Generator Weights Loaded')


def make_generator(
    state_dict_path: str, map_location: Union[int, str] = 'cuda', inference_mode: bool = True
) -> ProGenerator:
    gen_state_dict = torch.load(state_dict_path, map_location=map_location)
    gen_state_dict.update(
        {'settings': {'channel_depth': [512, 512, 512, 512, 256, 128, 64, 32, 16]}}
    )
    generator = ProGenerator()
    generator.load(gen_state_dict)

    if inference_mode:
        for p in generator.parameters():
            p.requires_grad = False
        generator.eval()

    return generator
