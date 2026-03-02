from __future__ import annotations
import torch
import torch.nn.functional as F
import time


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self, feature_matrix: torch.Tensor, targets: torch.Tensor, device: torch.device = None) -> None:
        if device is None:
            device = default_device()
        self.feature_matrix: torch.Tensor = feature_matrix.to(device)
        self.targets: torch.Tensor = targets.to(device)
        self.device: torch.device = device

    def __getitem__(self, idx):
        return self.feature_matrix[idx], self.targets[idx]

    def __len__(self) -> int:
        return self.feature_matrix.size(0)


class DataLoaderIterator:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool) -> None:
        self.shuffle: bool = shuffle
        self.max_considered_idx: int = (len(dataset) // batch_size) * batch_size - 1
        self.dataset: Dataset = dataset
        self.device: torch.device = dataset.device
        self._stop_next = False
        if self.shuffle:
            self.permutation: torch.Tensor = torch.randperm(len(dataset), device=self.device)
            # self.dataset = self.dataset[permutation]
        else:
            self.permutation: torch.Tensor = torch.arange(len(dataset), device=self.device)

        self.batch_size: int = batch_size
        self.position: int = 0

    def __iter__(self) -> DataLoaderIterator:
        return self

    def __next__(self):
        if self._stop_next:
            raise StopIteration
        idx = self.permutation[self.position:self.position + self.batch_size]
        self.position += self.batch_size

        x_batch, y_batch = self.dataset[idx]

        if self.position - 1 == self.max_considered_idx:
            self._stop_next = True
        return x_batch, y_batch


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle

    def __iter__(self) -> DataLoaderIterator:
        return DataLoaderIterator(self.dataset, self.batch_size, self.shuffle)


class Module:
    def __init__(self, training: bool = False, device: torch.device = None):
        self.training: bool = training
        self.device: torch.device = default_device() if device is None else device

    def __call__(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward(feature_matrix)

    def forward(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward is not defined.")

    def parameters(self) -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        for self_obj in self.__dict__.values():
            if isinstance(self_obj, torch.Tensor) and self_obj.requires_grad:
                params.append(self_obj)
            elif isinstance(self_obj, Module):
                params.extend(self_obj.parameters())
            elif isinstance(self_obj, list):
                for item in self_obj:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    else:
                        raise NotImplementedError("Only list of Modules are allowed.")
            else:
                # raise NotImplementedError("Only tensors, modules and lists of modules are allowed.")
                continue

        return params

    def to(self, device: torch.device) -> None:
        self.device = device
        for name, self_obj in self.__dict__.items():
            if isinstance(self_obj, torch.Tensor):
                requires_grad: bool = self_obj.requires_grad
                self.__dict__[name] = self_obj.detach().to(device).requires_grad_(requires_grad)
            elif isinstance(self_obj, Module):
                self_obj.to(device)
            elif isinstance(self_obj, (list, tuple)):
                for item in self_obj:
                    if isinstance(item, Module):
                        item.to(device)
                    else:
                        raise NotImplementedError("Only list of Modules are allowed.")
            else:
                # raise NotImplementedError("Only tensors, modules and lists of modules are allowed.")
                continue

    def train(self) -> None:
        self.training = True
        for obj in self.__dict__.values():
            if isinstance(obj, Module):
                obj.train()
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    if isinstance(item, Module):
                        item.train()
                    else:
                        raise NotImplementedError("This shouldn't have happened.")

    def eval(self) -> None:
        self.training = False
        for obj in self.__dict__.values():
            if isinstance(obj, Module):
                obj.eval()
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    if isinstance(item, Module):
                        item.eval()
                    else:
                        raise NotImplementedError("This shouldn't have happened.")


class Sequential(Module):
    def __init__(self, *layers, device=None):
        super().__init__(device=default_device() if device is None else device)
        self.layers = list(layers)
        if device is not None:
            for layer in self.layers:
                if isinstance(layer, Module):
                    layer.to(device)
                else:
                    raise TypeError("Sequential supports only Module layers")

    def forward(self, value_matrix: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value_matrix = layer(value_matrix)
        return value_matrix


class LinearLayer(Module):
    def __init__(self, input_size: int, output_size: int, device=None) -> None:
        super().__init__(device=default_device() if device is None else device)
        scale = (2.0 / input_size) ** 0.5
        weights = torch.randn(input_size, output_size, device=self.device) * scale
        self.weights = weights.detach().requires_grad_(True)
        biases = torch.zeros(output_size, device=self.device)
        self.biases = biases.detach().requires_grad_(True)

    def forward(self, input_matrix: torch.Tensor):
        return input_matrix @ self.weights + self.biases


class ReLU(Module):
    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.clamp(matrix, min=0)


class Conv2D(Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, device: torch.device = None) -> None:
        super().__init__(device=default_device() if device is None else device)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.padding = (self.kernel_size - 1) // 2

        self.weights: torch.Tensor = torch.rand(self.out_channels, self.in_channels, self.kernel_size,
                                                self.kernel_size) - 0.5
        self.biases: torch.Tensor = torch.rand(self.out_channels) - 0.5

        fan_in = in_channels * kernel_size * kernel_size
        scale = (2.0 / fan_in) ** 0.5
        weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=self.device) * scale
        self.weights = weights.detach().requires_grad_(True)
        biases = torch.zeros(out_channels, device=self.device)
        self.biases = biases.detach().requires_grad_(True)

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return F.conv2d(matrix, self.weights, self.biases, padding=self.padding)


class BatchNorm2D(Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, device=None):
        super().__init__(device=default_device() if device is None else device)
        self.num_channels: int = num_channels
        self.eps: float = eps
        gamma = torch.ones(1, num_channels, 1, 1, device=self.device)
        self.gamma = gamma.detach().requires_grad_(True)
        beta = torch.zeros(1, num_channels, 1, 1, device=self.device)
        self.beta = beta.detach().requires_grad_(True)
        self.running_mean = torch.zeros(1, num_channels, 1, 1, device=self.device)
        self.running_var = torch.ones(1, num_channels, 1, 1, device=self.device)
        self.momentum: float = 0.1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean: torch.Tensor = image.mean(dim=(0, 2, 3), keepdim=True)
            var: torch.Tensor = image.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean: torch.Tensor = self.running_mean
            var: torch.Tensor = self.running_var

        normilized_image: torch.Tensor = (image - mean) / torch.sqrt(var + self.eps)

        return self.gamma * normilized_image + self.beta


class Dropout(Module):
    def __init__(self, probability: float = 0.5, device=None):
        super().__init__(device=default_device() if device is None else device)
        self.probability: float = probability
        if self.probability == 1.0:
            raise ValueError

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return image
        mask: torch.Tensor = (torch.rand_like(image) > self.probability).float()
        return image * mask / (1.0 - self.probability)


class AvgPool2D(Module):
    def __init__(self, kernel_size: int = 2):
        self.kernel_size: int = kernel_size

    def forward(self, image: torch.Tensor):
        return F.avg_pool2d(image, kernel_size=self.kernel_size)


class Optimizer:
    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters: list[torch.Tensor], lr: float):
        self.lr: float = lr
        self.parameters: list[torch.Tensor] = parameters

    def step(self):
        with torch.no_grad():
            for weights in self.parameters:
                if weights.grad is not None:
                    weights -= self.lr * weights.grad

    def zero_grad(self):
        with torch.no_grad():
            for weights in self.parameters:
                if weights.grad is not None:
                    weights.grad.zero_()


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.size(0), device=logits.device)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_probs[rows, targets].mean()


def train(model: Module, dataset: Dataset, epochs: int, optimizer: Optimizer | None = None, batch_size: int = 32,
          shuffle: bool = True) -> None:
    device: torch.device = model.device
    if optimizer is None:
        model.to(device)
        optimizer = SGD(model.parameters(), lr=1e-3)
    model.train()
    accuracy: list[float] = []
    loss_function: list[float] = []
    for epoch in range(epochs):
        epoch_start_time: float = time.time()
        loader: DataLoader = DataLoader(dataset, batch_size, shuffle)
        for feature_batch, targets_batch in loader:
            feature_batch = feature_batch.to(model.device)
            targets_batch = targets_batch.to(model.device)

            optimizer.zero_grad()
            predictions_batch: torch.Tensor = model(feature_batch)
            loss: float = cross_entropy(predictions_batch, targets_batch)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(dataset.feature_matrix)
            _, predicted_classes = torch.max(logits, dim=1)
            accuracy.append((predicted_classes == dataset.targets).float().mean().item())
            loss_function.append(cross_entropy(logits, dataset.targets).item())
        epoch_end_time: float = time.time()
        epoch_train_time: float = epoch_end_time - epoch_start_time
        print(
            f"Epoch: {epoch + 1}, accuracy = {accuracy[-1]:.2f}, epoch train time = {round(epoch_train_time, 1)} seconds.")
    model.eval()
    return accuracy, loss_function