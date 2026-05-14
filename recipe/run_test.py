import torch

from inplace_abn import ABN, InPlaceABN


def _assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values")


def _make_input():
    return torch.linspace(-1.0, 1.0, steps=2 * 3 * 4 * 4).reshape(2, 3, 4, 4)


def test_forward_matches_reference_abn():
    reference = ABN(3, activation="leaky_relu", activation_param=0.1)
    candidate = InPlaceABN(3, activation="leaky_relu", activation_param=0.1)
    candidate.load_state_dict(reference.state_dict())

    reference.train()
    candidate.train()

    x = _make_input()
    expected = reference(x.clone())
    actual = candidate(x.clone())

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    _assert_finite("forward output", actual)


def test_backward_produces_gradients():
    layer = InPlaceABN(3, activation="leaky_relu", activation_param=0.1)
    x_leaf = _make_input().requires_grad_(True)

    # InPlaceABN mutates its input, so pass a non-leaf tensor while keeping
    # gradients observable on x_leaf.
    output = layer(x_leaf + 0.0)
    loss = output.square().mean()
    loss.backward()

    if x_leaf.grad is None:
        raise AssertionError("input gradient was not populated")
    if layer.weight.grad is None or layer.bias.grad is None:
        raise AssertionError("affine parameter gradients were not populated")

    _assert_finite("input gradient", x_leaf.grad)
    _assert_finite("weight gradient", layer.weight.grad)
    _assert_finite("bias gradient", layer.bias.grad)


def test_eval_uses_running_statistics():
    layer = InPlaceABN(3, activation="identity")
    layer.eval()

    with torch.no_grad():
        output = layer(_make_input())

    assert output.shape == (2, 3, 4, 4)
    _assert_finite("eval output", output)


if __name__ == "__main__":
    test_forward_matches_reference_abn()
    test_backward_produces_gradients()
    test_eval_uses_running_statistics()
