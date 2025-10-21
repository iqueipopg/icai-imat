"""
This script contains the code for the Guided Backpropagation.
"""

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from src.utils import normalize_tensor, saliency_map

class GuidedBackprop(nn.Module):
    """
    Computes Guided Backpropagation for model interpretability. It is IMPORTANT that the
    ReLU layers in the model are defined as torch.nn.ReLU instead of F.relu, otherwise
    hooks will not be registered correctly.

    Guided backpropagation computes the gradient of the target output with respect to
    the input, but gradients of ReLU functions are overridden so that only non-negative
    gradients are backpropagated.

    More details can be found in: [https://arxiv.org/abs/1412.6806](https://arxiv.org/abs/1412.6806)
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model
        self.activation_maps: list[torch.Tensor] = []
        self.hooks: list[RemovableHandle] = []

    def _register_hooks(self) -> None:
        """
        Modifies forward and backward hooks for the ReLU and registers it in the
        modules.

        Activation maps in each layer will be saved in the forward pass and deleted in
        the backward pass. This sequential structure helps follow perfectly the forward
        and backward passes: layer_1 -> ... -> layer_n (forward),
        layer_n -> ... -> layer_1 (backward).
        """

        def backward_hook(
            module: nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor]:
            """
            Backward hooks.

            Args:
                module: Module of the hook.
                grad_in: Input gradient.
                grad_out: Output gradient.

            Returns:
                Output gradient.
            """
            # TODO:
            # Solo deja pasar gradientes positivos
            return (torch.clamp(grad_in[0], min=0.0),)

        def forward_hook(
            module: nn.Module, inp: torch.Tensor, out: torch.Tensor
        ) -> None:
            """
            Forward hooks. We will not use this, but if we wanted to extract the
            activations of each layer, for example to visualize them, we could use this.

            Args:
                module: Module of the hook.
                inp: Input tensor.
                out: Output tensor.

            Returns:
                Output tensor.
            """
            # TODO:
            self.activation_maps.append(out.data)

        # Registra hooks en todas las capas ReLU
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                handle_fwd = module.register_forward_hook(forward_hook)
                handle_bwd = module.register_backward_hook(backward_hook)
                self.hooks.append(handle_fwd)
                self.hooks.append(handle_bwd)

    def _delete_hooks_and_activation_maps(self) -> None:
        """
        Deletes all registered hooks.
        """
        # TODO:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activation_maps.clear()

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.

        Returns:
            Explanation. Dimensions: [batch, height, width].

        Raises:
            RuntimeError: If no ReLU modules are detected.
        """
        # TODO:
        self.model.eval()
        self._register_hooks()

        # VERIFICACIÓN: Si no se ha registrado ningún hook, lanzar error
        if len(self.hooks) == 0:
            self._delete_hooks_and_activation_maps()
            raise RuntimeError("No ReLU modules detected. GuidedBackprop requires nn.ReLU layers.")

        x = x.clone().detach().requires_grad_(True)

        output = self.model(x)
        if target_class is None:
            idx = output.argmax(dim=1)
        else:
            idx = torch.full((x.shape[0],), target_class, dtype=torch.long, device=x.device)
        score = output[torch.arange(x.shape[0]), idx]
        self.model.zero_grad()
        score.sum().backward()

        result = x.grad
        # Limpieza siempre al final
        self._delete_hooks_and_activation_maps()

        if result is None:
            raise RuntimeError("No gradients found. Are there ReLU layers defined as nn.ReLU?")

        explanation = result.abs().max(dim=1)[0]
        explanation = normalize_tensor(explanation)

        return explanation
