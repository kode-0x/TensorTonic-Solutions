import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    trained_weights = [w.clone() for w in weights]
    trained_biases = [b.clone() for b in biases]
    loss_history = []

    def forward():
        h = inputs
        for w, b in zip(trained_weights, trained_biases):
            h = torch.tanh(h @ w.T + b)
        return h.squeeze(-1)

    def compute_loss(predictions):
        return torch.sum((predictions - targets) ** 2)

    for _ in range(steps):
        activations = [inputs]
        h = inputs

        for w, b in zip(trained_weights, trained_biases):
            h = torch.tanh(h @ w.T + b)
            activations.append(h)

        predictions = h.squeeze(-1)
        loss = compute_loss(predictions)
        loss_history.append(loss.clone())

        if inputs.shape[0] == 0:
            continue

        delta = 2 * (predictions - targets).unsqueeze(-1)

        weight_grads = [None] * len(trained_weights)
        bias_grads = [None] * len(trained_biases)

        for layer in range(len(trained_weights) - 1, -1, -1):
            a = activations[layer + 1]
            delta = delta * (1 - a * a)

            weight_grads[layer] = delta.T @ activations[layer]
            bias_grads[layer] = delta.sum(dim=0)

            if layer > 0:
                delta = delta @ trained_weights[layer]

        trained_weights = [
            w - learning_rate * grad
            for w, grad in zip(trained_weights, weight_grads)
        ]
        trained_biases = [
            b - learning_rate * grad
            for b, grad in zip(trained_biases, bias_grads)
        ]

    predictions = forward()
    loss = compute_loss(predictions)

    return predictions, loss, trained_weights, trained_biases, loss_history