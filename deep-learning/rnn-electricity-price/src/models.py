# deep learning libraries
import torch
import torch.nn as nn

# other libraries
import math
from typing import Any


class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # TODO
        batch_size, seq_len, _ = inputs.shape
        hidden_size = h0.shape[2]

        # Inicializar el tensor de salida
        outputs = torch.zeros(
            batch_size, seq_len, hidden_size, device=inputs.device
        ).double()
        h_t = h0.squeeze(0)

        for t in range(seq_len):
            h_t = (
                torch.matmul(inputs[:, t], weight_ih.T)
                + bias_ih
                + torch.matmul(h_t, weight_hh.T)
                + bias_hh
            )
            h_t[h_t <= 0] = 0  # ReLU

            outputs[:, t] = h_t  # Guardamos salida sin activacion

        ctx.save_for_backward(
            inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh, outputs
        )

        return outputs, h_t.unsqueeze(0)

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """

        # TODO
        inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh, outputs = ctx.saved_tensors
        batch_size, seq_len, _ = inputs.shape

        grad_inputs = torch.zeros_like(inputs)
        grad_h0 = torch.zeros_like(h0)
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_bias_ih = torch.zeros_like(bias_ih)
        grad_bias_hh = torch.zeros_like(bias_hh)

        grad_h_t = grad_hn.squeeze(0)

        for t in reversed(range(seq_len)):
            grad_h_t = grad_h_t + grad_output[:, t]  # Acumulamos gradientes

            # Aplicar la derivada de ReLU: ReLU' = 1 si h > 0, 0 si h <= 0
            grad_h_t *= (outputs[:, t] > 0).float()

            # Gradientes de los pesos y biases
            grad_weight_ih += grad_h_t.T @ inputs[:, t]
            if t > 0:
                grad_weight_hh += grad_h_t.T @ outputs[:, t - 1]
            else:
                grad_weight_hh += grad_h_t.T @ h0.squeeze(0)

            grad_bias_ih += grad_h_t.sum(dim=0)
            grad_bias_hh += grad_h_t.sum(dim=0)

            # Gradientes respecto a la entrada
            grad_inputs[:, t] = grad_h_t @ weight_ih

            # Propagamos el gradiente hacia el estado oculto anterior
            grad_h_t = grad_h_t @ weight_hh

        # Guardamos el gradiente final de h0
        grad_h0 = grad_h_t.unsqueeze(0)

        return (
            grad_inputs,
            grad_h0,
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_ih,
            grad_bias_hh,
        )


class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        This method is the constructor of the RNN layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))

        # init parameters corectly
        self.reset_parameters()

        self.fn = RNNFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

        return None


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
        """

        # TODO
        super().__init__()
        self.hidden_size: int = hidden_size
        self.input_size: int = 24
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=5,
            batch_first=True,
            dropout=0.4,
        )
        self.fc = nn.Linear(self.hidden_size, 24)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """

        # TODO
        inputs = inputs.float()
        output, (h_t, c_n) = self.lstm(inputs)
        h_n = h_t[-1]
        return self.fc(h_n)
