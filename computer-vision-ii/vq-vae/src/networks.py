"""
Encoder and Decoder architectures for VQ-VAE - Incomplete version.

Students need to implement the network architectures and forward passes.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    """Residual block with skip connection.
    
    A residual block applies a sequence of operations and adds the result
    to the input (skip connection). This helps with gradient flow in deep networks.
    """

    def __init__(
        self, in_channels: int, num_hiddens: int, num_residual_hiddens: int
    ) -> None:
        """Initialize residual block.
        
        The block should contain:
        1. ReLU activation
        2. 3x3 convolution (in_channels -> num_residual_hiddens)
        3. ReLU activation  
        4. 1x1 convolution (num_residual_hiddens -> num_hiddens)
        
        Args:
            in_channels: Number of input channels
            num_hiddens: Number of output channels (should equal in_channels for residual)
            num_residual_hiddens: Number of hidden channels in the block
        """
        super().__init__()
        #[ ] Create sequential block with ReLU and Conv2d layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection applied
        """
        #[ ] Apply block with residual connection
        pass


class ResidualStack(nn.Module):
    """Stack of residual blocks."""

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        """Initialize stack of residual blocks.
        
        Args:
            in_channels: Number of input channels
            num_hiddens: Number of hidden channels
            num_residual_layers: Number of residual blocks to stack
            num_residual_hiddens: Hidden channels in each residual block
        """
        super().__init__()
        #[ ] Create a ModuleList of residual blocks
        # Each block should have the same parameters
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual stack.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all residual blocks and final ReLU
        """
        #[ ] Pass through residual layers and apply final ReLU
        pass


class Encoder(nn.Module):
    """VQ-VAE encoder with convolutional layers and residual blocks.
    
    The encoder progressively downsamples the input image while increasing
    the number of channels, ending with residual blocks for feature refinement.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        """Initialize encoder.
        
        The encoder should have:
        1. Conv2d: in_channels -> num_hiddens//2, kernel=4, stride=2, padding=1
        2. Conv2d: num_hiddens//2 -> num_hiddens, kernel=4, stride=2, padding=1  
        3. Conv2d: num_hiddens -> num_hiddens, kernel=3, stride=1, padding=1
        4. ResidualStack with the given parameters
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            num_hiddens: Number of hidden channels
            num_residual_layers: Number of residual layers
            num_residual_hiddens: Hidden channels in residual blocks
        """
        super().__init__()
        #[ ] Initialize the three convolutional layers as described above
        
        #[ ] Initialize the residual stack
        pass

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            inputs: Input images [B, C, H, W]
            
        Returns:
            Encoded latent representation [B, num_hiddens, H//4, W//4]
        """
        #[ ] Apply convolutions and residual stack
        pass


class Decoder(nn.Module):
    """VQ-VAE decoder with transposed convolutions and residual blocks.
    
    The decoder upsamples the quantized latent representation back to
    the original image resolution.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        """Initialize decoder.
        
        The decoder should have:
        1. Conv2d: in_channels -> num_hiddens, kernel=3, stride=1, padding=1
        2. ResidualStack with the given parameters
        3. ConvTranspose2d: num_hiddens -> num_hiddens//2, kernel=4, stride=2, padding=1
        4. ConvTranspose2d: num_hiddens//2 -> 3, kernel=4, stride=2, padding=1
        
        Args:
            in_channels: Number of input channels (embedding_dim)
            num_hiddens: Number of hidden channels
            num_residual_layers: Number of residual layers
            num_residual_hiddens: Hidden channels in residual blocks
        """
        super().__init__()
        #[ ] Initialize convolution, residual stack, and transposed convolutions
        pass

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation to reconstruction.
        
        Args:
            inputs: Quantized latent representation [B, embedding_dim, H, W]
            
        Returns:
            Reconstructed images [B, 3, H*4, W*4]
        """
        #[ ] Apply convolution, residual stack, and transposed convolutions
        pass