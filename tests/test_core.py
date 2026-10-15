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


def test_simulate_uses_the_exact_martingale_drift():
    sigma, s0 = 0.2, -0.1
    s = core.simulate(1000, 28, sigma, s0, torch.Generator().manual_seed(0))
    z = torch.randn(28, 1000, generator=torch.Generator().manual_seed(0))
    expected = s0 + torch.cumsum(sigma * math.sqrt(core.DT) * z - 0.5 * sigma**2 * core.DT, 0)
    assert s.shape == (29, 1000)
    assert torch.all(s[0] == s0)
    assert torch.allclose(s[1:], expected, atol=1e-6)


def test_pnl_on_a_hand_written_path():
    s = torch.log(torch.tensor([[1.0], [1.1], [0.9], [1.2]]))  # European, K = 1
    zero = core.pnl(lambda *a: torch.zeros(1), "european", s, 0.01, 0.2)
    assert zero.item() == pytest.approx(-0.2, abs=1e-6)  # minus the payoff
    one = core.pnl(lambda *a: torch.ones(1), "european", s, 0.01, 0.2)
    assert one.item() == pytest.approx(1.2 - 1.0 - 0.01 * 1.0 - 0.2, abs=1e-6)  # cost only on the first trade


def test_lookback_payoff_includes_the_past_maximum():
    s = torch.log(torch.tensor([[1.0], [1.01], [0.99]]) / 1.03)
    past_max = math.log(1.10 / 1.03)
    wealth = core.pnl(lambda *a: torch.zeros(1), "lookback", s, 0.0, 0.2, m0=past_max)
    assert wealth.item() == pytest.approx(-(1.10 - 1.03), abs=1e-6)


def test_entropic_price_and_utility_match_reference():
    wealth = -torch.arange(4.0)
    assert core.utility(wealth) == pytest.approx(-7.7982, abs=1e-4)
    assert core.price(wealth) == pytest.approx(2.0539, abs=1e-4)


def test_whalley_wilmott_half_width():
    s, m, tau = t64(0.0, 0.0, 15 / 365)
    _, _, gamma_k = core.european(s, tau, 0.2)
    lo, hi = core.ww_band("european", 1e-3, s, m, tau, 0.2)
    assert ((hi - lo) / 2).item() == pytest.approx((1.5 * 1e-3 * gamma_k.item() ** 2) ** (1 / 3), rel=1e-12)
    lo0, hi0 = core.ww_band("european", 0.0, s, m, tau, 0.2)
    assert lo0.item() == hi0.item()


def no_hedge(s, m, tau, sigma, prev):
    return torch.zeros_like(s)


@pytest.mark.parametrize(
    "derivative,policy,expected,tol",
    [
        ("european", core.whalley_wilmott("european", 0.0), 0.022101, 3e-5),
        ("lookback", core.whalley_wilmott("lookback", 0.0), 0.01804, 6e-5),
        ("european", no_hedge, 0.022673, 2.4e-4),
        ("lookback", no_hedge, 0.018408, 2.0e-4),
    ],
)
def test_paper_anchors(derivative, policy, expected, tol):
    # zero-cost delta hedge vs Tables B2/B4 c=0 rows; no-hedge prices measured with 10M paths (28 daily steps)
    s = core.simulate(200_000, 28, 0.2, -math.log(core.STRIKE[derivative]), torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert core.price(core.pnl(policy, derivative, s, 0.0, 0.2)) == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize(
    "derivative,method,n_params",
    [("european", "ntb", 3362), ("lookback", "ntb", 3394), ("european", "ffn", 3361), ("lookback", "ffn", 3393)],
)
def test_network_sizes_match_the_paper(derivative, method, n_params):
    assert sum(p.numel() for p in core.Hedger(derivative, method).parameters()) == n_params


def test_ntb_band_is_centred_on_delta_and_holds_a_hedge_inside_it():
    torch.manual_seed(0)
    net = core.Hedger("lookback", "ntb")
    s = torch.linspace(-0.1, 0.05, 50)
    m = s.clamp(min=-0.03)
    tau = torch.tensor(10 / 365)
    lo, hi = net.band(s, m, tau, 0.2)
    # 1. Band is never inverted with this seed
    assert torch.all(lo < hi)
    # 2. No trade inside the band (hedge stays at entry point)
    q = lo + 0.25 * (hi - lo)
    assert torch.equal(net(s, m, tau, 0.2, q), q)
    # 3. Clamping: hedge below band -> lo, hedge above band -> hi
    assert torch.equal(net(s, m, tau, 0.2, torch.full_like(s, -5.0)), lo)
    assert torch.equal(net(s, m, tau, 0.2, torch.full_like(s, 5.0)), hi)
    # 4. Centring: zero last layer -> band offsets are zero -> clamp(prev, delta, delta) = delta
    with torch.no_grad():
        net.net[-1].weight.zero_()
        net.net[-1].bias.zero_()
    delta = core.greeks("lookback", s, m, tau, 0.2)[1]
    assert torch.allclose(net(s, m, tau, 0.2, torch.full_like(s, 0.3)), delta)


def test_ffn_depends_on_the_previous_hedge():
    torch.manual_seed(0)
    net = core.Hedger("european", "ffn")
    s, tau = torch.zeros(4), torch.tensor(10 / 365)
    assert not torch.equal(net(s, s, tau, 0.2, torch.zeros(4)), net(s, s, tau, 0.2, torch.ones(4)))


def test_pnl_charges_cost_on_every_trade_from_the_previous_hedge():
    s = torch.log(torch.tensor([[1.0], [1.1], [0.9], [1.2]]))  # European, K = 1, payoff 0.2
    hedges = iter([torch.tensor([1.0]), torch.tensor([0.4]), torch.tensor([0.9])])  # buy, sell, buy
    wealth = core.pnl(lambda *a: next(hedges), "european", s, 0.01, 0.2)
    gains = 1.0 * 0.1 + 0.4 * -0.2 + 0.9 * 0.3
    costs = 0.01 * (1.0 * 1.0 + 1.1 * 0.6 + 0.9 * 0.5)  # c * S_i * |h_i - h_{i-1}|
    assert wealth.item() == pytest.approx(gains - costs - 0.2, abs=1e-6)
    held = core.pnl(lambda *a: torch.ones(1), "european", s, 0.01, 0.2, prev0=1.0)
    assert held.item() == pytest.approx(1.2 - 1.0 - 0.2, abs=1e-6)  # already hedged: no trade, no cost
