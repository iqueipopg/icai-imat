# deep learning libraries
import torch

# other libraries
from typing import Iterator, Dict, Any, DefaultDict


class SGD(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self, params: Iterator[torch.nn.Parameter], lr=1e-3, weight_decay: float = 0.0
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay)

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        # Recorremos cada conjunto de hiperparámetros
        for config in self.param_groups:
            # Extraemos la tasa de aprendizaje y el factor de regularización
            learning_rate = config["lr"]
            reg_factor = config["weight_decay"]

            # Procesamos cada tensor de parámetros en el grupo actual
            for tensor in config["params"]:
                if tensor.grad is None:
                    # Si el tensor no tiene gradiente, lo ignoramos
                    continue

                grad_tensor = (
                    tensor.grad.data.clone()
                )  # Copiamos el gradiente para evitar modificarlo directamente

                # Si hay regularización, aplicamos la penalización L2
                if reg_factor != 0:
                    decay_contrib = torch.zeros_like(
                        tensor.data
                    )  # Creamos un tensor para la contribución de la regularización
                    decay_flat = decay_contrib.view(
                        -1
                    )  # Aplanamos para iterar más fácilmente
                    tensor_flat = tensor.data.view(-1)  # Aplanamos los datos originales

                    # Calculamos la penalización L2 para cada elemento del tensor
                    for i in range(len(decay_flat)):
                        decay_flat[i] = tensor_flat[i] * reg_factor

                    grad_flat = grad_tensor.view(-1)  # Aplanamos el gradiente
                    for i in range(len(grad_flat)):
                        grad_flat[i] += decay_flat[
                            i
                        ]  # Sumamos la regularización al gradiente

                update_tensor = torch.zeros_like(
                    grad_tensor
                )  # Creamos un tensor vacío para almacenar los cambios
                update_flat = update_tensor.view(-1)  # Aplanamos
                grad_flat = grad_tensor.view(-1)  # Reutilizamos el gradiente aplanado

                for i in range(len(update_flat)):
                    update_flat[i] = (
                        grad_flat[i] * learning_rate
                    )  # Multiplicamos por la tasa de aprendizaje

                # Creamos un tensor nuevo para los valores actualizados
                new_values = torch.zeros_like(tensor.data)
                new_values_flat = new_values.view(-1)
                tensor_flat = tensor.data.view(-1)

                # Aplicamos la actualización a cada elemento del tensor
                for i in range(len(new_values_flat)):
                    new_values_flat[i] = tensor_flat[i] - update_flat[i]

                # Copiamos los nuevos valores en el tensor original
                tensor.data.copy_(new_values)


class SGDMomentum(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Attr:
            param_groups: list with the dict of the parameters.
            state: dict with the state for each parameter.
        """
        # TODO
        # Iteramos sobre cada conjunto de hiperparámetros definidos en la configuración
        for config in self.param_groups:
            # Extraemos la tasa de aprendizaje, el momentum y el peso de regularización
            learning_rate = config["lr"]
            momentum_factor = config["momentum"]
            reg_strength = config["weight_decay"]

            # Procesamos cada tensor de parámetros en el grupo actual
            for tensor in config["params"]:
                if tensor.grad is None:
                    # Ignoramos los tensores sin gradiente
                    continue

                # Copiamos el gradiente para evitar modificar el original directamente
                gradient = tensor.grad.data.clone()

                # Si hay regularización L2, aplicamos la penalización correspondiente
                if reg_strength != 0:

                    decay_component = torch.zeros_like(tensor.data)
                    decay_flat = decay_component.view(-1)
                    tensor_flat = tensor.data.view(-1)

                    # Calculamos la penalización L2 (λ * θ) elemento a elemento
                    for i in range(len(decay_flat)):
                        decay_flat[i] = tensor_flat[i] * reg_strength

                    grad_flat = gradient.view(-1)  # Aplanamos el gradiente
                    for i in range(len(grad_flat)):
                        grad_flat[i] += decay_flat[
                            i
                        ]  # Sumamos el término de weight decay

                # Accedemos al estado del optimizador para este tensor
                optimizer_state = self.state[tensor]

                # Si el buffer de momentum no existe, lo inicializamos con ceros
                if "momentum_cache" not in optimizer_state:
                    momentum_cache = torch.zeros_like(tensor.data)
                    optimizer_state["momentum_cache"] = momentum_cache

                # Recuperamos el buffer de momentum
                momentum_cache = optimizer_state["momentum_cache"]

                # Aplicamos la actualización de momentum: v = μ * v + g
                buffer_flat = momentum_cache.view(-1)
                for i in range(len(buffer_flat)):
                    buffer_flat[
                        i
                    ] *= momentum_factor  # Multiplicamos por el coeficiente de momentum

                # Sumamos el gradiente al buffer de momentum
                grad_flat = gradient.view(-1)
                for i in range(len(buffer_flat)):
                    buffer_flat[i] += grad_flat[i]

                # Calculamos la actualización de los parámetros: θ = θ - lr * v
                update_values = torch.zeros_like(momentum_cache)
                update_flat = update_values.view(-1)
                buffer_flat = momentum_cache.view(-1)

                for i in range(len(update_flat)):
                    update_flat[i] = (
                        buffer_flat[i] * learning_rate
                    )  # Multiplicamos por la tasa de aprendizaje

                # Creamos un nuevo tensor para los valores corregidos de los parámetros
                updated_params = torch.zeros_like(tensor.data)
                updated_flat = updated_params.view(-1)
                tensor_flat = tensor.data.view(-1)

                # Restamos la actualización de los valores anteriores del tensor
                for i in range(len(updated_flat)):
                    updated_flat[i] = tensor_flat[i] - update_flat[i]

                # Aplicamos la actualización a los parámetros del tensor
                tensor.data.copy_(updated_params)


class SGDNesterov(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        # Definimos los valores por defecto para este optimizador
        defaults: Dict[Any, Any] = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum
        )

        # Llamamos al constructor de la clase padre
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        # Iteramos sobre cada conjunto de parámetros configurados en el optimizador
        for param_set in self.param_groups:
            # Extraemos la tasa de aprendizaje, el peso de regularización y el momentum
            learning_rate = param_set["lr"]
            reg_factor = param_set["weight_decay"]
            momentum_coeff = param_set["momentum"]

            # Recorremos cada tensor de parámetros dentro del grupo
            for tensor in param_set["params"]:
                if tensor.grad is None:
                    # Omitimos los tensores que no tienen gradiente calculado
                    continue

                # Creamos una copia del gradiente para evitar modificaciones no deseadas
                gradient = tensor.grad.data.clone()

                # Si hay regularización L2 (Weight Decay), ajustamos el gradiente
                if reg_factor != 0:
                    # Aplicamos el término de weight decay: g_t = g_t + λθ_t
                    for i in range(gradient.numel()):
                        gradient.view(-1)[i] += reg_factor * tensor.data.view(-1)[i]

                # Accedemos al estado interno del optimizador para este tensor
                tensor_state = self.state[tensor]

                if "velocity_buffer" not in tensor_state:
                    tensor_state["velocity_buffer"] = gradient.clone()
                else:
                    # actualización del momentum: v_t = μ*v_{t-1} + g_t
                    for i in range(gradient.numel()):
                        tensor_state["velocity_buffer"].view(-1)[i] = (
                            tensor_state["velocity_buffer"].view(-1)[i] * momentum_coeff
                            + gradient.view(-1)[i]
                        )

                # Aplicamos la corrección de Nesterov: g_t = g_t + μ*v_t

                for i in range(gradient.numel()):
                    gradient.view(-1)[i] += (
                        momentum_coeff * tensor_state["velocity_buffer"].view(-1)[i]
                    )

                #
                for i in range(tensor.data.numel()):
                    tensor.data.view(-1)[i] -= learning_rate * gradient.view(-1)[i]


class Adam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        # Definimos los valores por defecto
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # Llamamos al constructor de la clase padre
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        # Recorremos cada conjunto de parámetros definido en el optimizador
        for param_group in self.param_groups:

            learning_rate = param_group["lr"]
            beta_m, beta_v = param_group["betas"]
            epsilon = param_group["eps"]
            decay_factor = param_group["weight_decay"]

            # Iteramos sobre cada tensor de parámetros dentro del grupo
            for weight_tensor in param_group["params"]:
                if weight_tensor.grad is None:
                    # Saltamos los parámetros sin gradiente
                    continue

                # Clonamos el gradiente actual para evitar modificaciones inesperadas
                gradient = weight_tensor.grad.data.clone()

                # Aplicamos la regularización L2 (Weight Decay) si es necesario
                if decay_factor != 0:
                    for i in range(gradient.numel()):
                        gradient.view(-1)[i] += (
                            decay_factor * weight_tensor.data.view(-1)[i]
                        )

                # Obtenemos el estado del optimizador para este tensor
                tensor_state = self.state[weight_tensor]

                # Si el estado aún no existe, inicializamos las variables necesarias
                if len(tensor_state) == 0:
                    tensor_state["timestep"] = 0  # Contador de iteraciones
                    tensor_state["momentum_avg"] = torch.zeros_like(
                        weight_tensor.data
                    )  # Estimación del primer momento (media)
                    tensor_state["velocity_avg"] = torch.zeros_like(
                        weight_tensor.data
                    )  # Estimación del segundo momento (varianza)

                # Accedemos a las estimaciones del primer y segundo momento
                momentum_avg = tensor_state["momentum_avg"]
                velocity_avg = tensor_state["velocity_avg"]

                # Aumentamos el contador de iteraciones
                tensor_state["timestep"] += 1
                timestep = tensor_state["timestep"]

                for i in range(momentum_avg.numel()):
                    momentum_avg.view(-1)[i] = (
                        beta_m * momentum_avg.view(-1)[i]
                        + (1 - beta_m) * gradient.view(-1)[i]
                    )

                for i in range(velocity_avg.numel()):
                    velocity_avg.view(-1)[i] = beta_v * velocity_avg.view(-1)[i] + (
                        1 - beta_v
                    ) * (gradient.view(-1)[i] ** 2)

                bias_correction_m = 1 - beta_m**timestep
                bias_correction_v = 1 - beta_v**timestep

                # Ajustamos la tasa de aprendizaje con la corrección de bias
                adjusted_step = learning_rate / bias_correction_m

                for i in range(weight_tensor.data.numel()):
                    # Corregimos el sesgo en la estimación del primer momento
                    corrected_momentum = momentum_avg.view(-1)[i]

                    # Corregimos el sesgo en la estimación del segundo momento
                    corrected_velocity = velocity_avg.view(-1)[i] / bias_correction_v

                    # Aplicamos la actualización del parámetro usando Adam
                    weight_tensor.data.view(-1)[i] -= (
                        adjusted_step
                        * corrected_momentum
                        / ((corrected_velocity**0.5) + epsilon)
                    )
