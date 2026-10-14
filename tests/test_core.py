import math

import pytest
import torch

from deep_hedging import core


def t64(*xs):
    return [torch.tensor(x, dtype=torch.float64) for x in xs]


def test_european_greeks_match_reference_and_closed_form():
    s, tau = t64([-0.01, 0.0, 0.01], 0.1)
    _, delta, _ = core.european(s, tau, 0.2)
    assert torch.allclose(delta, torch.tensor([0.4497, 0.5126, 0.5752], dtype=torch.float64), atol=1e-4)
    price, _, _ = core.european(*t64(0.0, 28 / 365), 0.2)
    assert price.item() == pytest.approx(0.0220962, abs=1e-7)


@pytest.mark.parametrize(
    "S,M,K,days,expected",
    [
        (1.0, 1.0, 1.03, 30, (0.0228074, 0.637942, 12.6734)),
        (1.02, 1.05, 1.03, 15, (0.0318061, 0.493096, 15.3960)),
        (1.05, 1.05, 1.03, 10, (0.0480230, 1.026689, 23.4338)),
    ],
)
def test_lookback_greeks_match_verified_vectors(S, M, K, days, expected):
    s, m, tau = t64(math.log(S / K), math.log(M / K), days / 365)
    price_k, delta, gamma_k = core.lookback(s, m, tau, 0.2)
    assert K * price_k.item() == pytest.approx(expected[0], abs=5e-7)
    assert delta.item() == pytest.approx(expected[1], abs=5e-7)
    assert gamma_k.item() / K == pytest.approx(expected[2], rel=1e-4)


@pytest.mark.parametrize("m_over_k", [0.98, 1.04])
def test_lookback_delta_and_gamma_are_derivatives_of_the_price(m_over_k):
    # away from S == M, analytic greeks must equal autograd of the closed-form price (in absolute units, K = 1.03)
    K = 1.03
    spot = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    m, tau = t64(math.log(m_over_k * 1.05 / K), 20 / 365)
    price_k, delta, gamma_k = core.lookback(torch.log(spot / K), m, tau, 0.2)
    (d_auto,) = torch.autograd.grad(K * price_k, spot, create_graph=True)
    (g_auto,) = torch.autograd.grad(d_auto, spot)
    assert d_auto.item() == pytest.approx(delta.item(), abs=1e-10)
    assert g_auto.item() == pytest.approx(gamma_k.item() / K, rel=1e-8)


def test_clamp_matches_reference_including_inverted_band():
    x = torch.linspace(-2, 12, 15) * 0.1
    assert torch.equal(core.clamp(x, torch.tensor(0.0), torch.tensor(1.0)), x.clamp(0.0, 1.0))
    out = core.clamp(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor(0.0))
    assert torch.equal(out, torch.tensor([0.0, 0.5]))
