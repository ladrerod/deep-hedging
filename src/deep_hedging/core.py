"""The paper's math: Black-Scholes greeks, hedging strategies, paths, P&L and the entropic price.

Units follow the paper: spot starts at 1, strike K = STRIKE[derivative], risk aversion 1 on absolute P&L.
Greeks return (price/K, delta, gamma*K), so everything is scale-free in log-moneyness s = log(S/K).
"""

import math

import torch
from torch import nn

DT = 1 / 365
STRIKE = {"european": 1.0, "lookback": 1.03}


def _ncdf(x):
    return torch.special.ndtr(x)


def _npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def european(s, tau, sigma):
    """Black-Scholes European call, zero rates. Returns (price/K, delta, gamma*K)."""
    a = sigma * torch.sqrt(tau)
    d1 = s / a + 0.5 * a
    es = torch.exp(s)
    return es * _ncdf(d1) - _ncdf(d1 - a), _ncdf(d1), _npdf(d1) / (es * a)


def lookback(s, m, tau, sigma):
    """Fixed-strike lookback call, zero rates, continuous monitoring; m = log(running max / K).

    Returns (price/K, delta, gamma*K). At S == max the gamma is the one-sided value (S below the max);
    autograd through max() returns gamma/4 there on torch 2.x, hence the closed form.
    """
    h = torch.maximum(m, s).clamp(min=0.0)
    a = sigma * torch.sqrt(tau)
    d1 = (s - h) / a + 0.5 * a
    n1, p1 = _ncdf(d1), _npdf(d1)
    g = d1 * n1 + p1
    es, eh = torch.exp(s), torch.exp(h)
    price = (eh - 1.0) + es * n1 - eh * _ncdf(d1 - a) + es * a * g
    return price, 2.0 * n1 + a * g, (2.0 * p1 / a + n1) / es


def greeks(derivative, s, m, tau, sigma):
    return european(s, tau, sigma) if derivative == "european" else lookback(s, m, tau, sigma)


def clamp(x, lo, hi):
    """Paper eq. 13; an inverted band (lo > hi) gives the midpoint."""
    return torch.where(lo < hi, torch.minimum(torch.maximum(x, lo), hi), (lo + hi) / 2)


def simulate(n_paths, days, sigma, s0=0.0, generator=None):
    """Driftless GBM in log-moneyness, shape (days + 1, n_paths), exact per-step drift -sigma^2 dt / 2."""
    z = torch.randn(days, n_paths, generator=generator)
    steps = sigma * math.sqrt(DT) * z - 0.5 * sigma**2 * DT
    return torch.cat([torch.full((1, n_paths), float(s0)), s0 + torch.cumsum(steps, 0)])


def pnl(policy, derivative, s, cost, sigma, prev0=0.0, m0=None):
    """Seller's final wealth (paper eq. 4) on paths s of shape (n + 1, N), in units where K = STRIKE[derivative].

    policy(s_i, m_i, tau_i, sigma, prev) returns the hedge held over [t_i, t_{i+1}]. Each trade costs
    cost * S_i * |change|; liquidation at maturity is free. m0 is a running maximum observed before s[0].
    """
    k = STRIKE[derivative]
    n = s.shape[0] - 1
    m = torch.cummax(s, 0).values
    if m0 is not None:
        m = torch.clamp(m, min=float(m0))
    spot = k * torch.exp(s)
    prev = torch.full_like(s[0], float(prev0))
    wealth = torch.zeros_like(s[0])
    for i in range(n):
        hedge = policy(s[i], m[i], torch.tensor((n - i) * DT), sigma, prev)
        wealth = wealth + hedge * (spot[i + 1] - spot[i]) - cost * spot[i] * (hedge - prev).abs()
        prev = hedge
    payoff = spot[-1] - k if derivative == "european" else k * torch.exp(m[-1]) - k
    return wealth - payoff.clamp(min=0.0)


def price(wealth, lam=1.0):
    """Entropic risk measure = indifference price: log E[exp(-lam P)] / lam, reduced in float64."""
    return math.log(torch.exp(-lam * wealth.double()).mean().item()) / lam


def utility(wealth, lam=1.0):
    """Expected exponential utility E[-exp(-lam P)]."""
    return -torch.exp(-lam * wealth.double()).mean().item()


def ww_band(derivative, cost, s, m, tau, sigma, lam=1.0):
    """Whalley-Wilmott asymptotic band (paper eq. 20): delta -/+ (1.5 c S gamma^2 / lam)^(1/3)."""
    _, delta, gamma_k = greeks(derivative, s, m, tau, sigma)
    w = (1.5 * cost * torch.exp(s) * gamma_k**2 / (STRIKE[derivative] * lam)) ** (1 / 3)
    return delta - w, delta + w


def whalley_wilmott(derivative, cost):
    """Hedging policy that keeps the hedge inside the Whalley-Wilmott band (a pure delta hedge at zero cost)."""
    return lambda s, m, tau, sigma, prev: clamp(prev, *ww_band(derivative, cost, s, m, tau, sigma))


def mlp(n_in, n_out):
    """4 hidden layers x 32 units, ReLU (paper Section 5)."""
    layers = []
    for i in range(4):
        layers += [nn.Linear(n_in if i == 0 else 32, 32), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(32, n_out))


class Hedger(nn.Module):
    """'ntb': no-transaction band network (paper eq. 12). 'ffn': feed-forward baseline (eq. 9)."""

    def __init__(self, derivative, method):
        super().__init__()
        self.derivative, self.method = derivative, method
        n_in = 3 if derivative == "european" else 4
        self.net = mlp(n_in, 2) if method == "ntb" else mlp(n_in + 1, 1)

    def features(self, s, m, tau, sigma):
        cols = [s] if self.derivative == "european" else [s, m]
        return torch.stack([*cols, tau.expand_as(s), torch.full_like(s, sigma)], -1)

    def band(self, s, m, tau, sigma):
        delta = greeks(self.derivative, s, m, tau, sigma)[1]
        o = nn.functional.leaky_relu(self.net(self.features(s, m, tau, sigma)), 0.01)
        return delta - o[..., 0], delta + o[..., 1]

    def forward(self, s, m, tau, sigma, prev):
        if self.method == "ntb":
            return clamp(prev, *self.band(s, m, tau, sigma))
        delta = greeks(self.derivative, s, m, tau, sigma)[1]
        x = torch.cat([self.features(s, m, tau, sigma), prev.unsqueeze(-1)], -1)
        return delta + torch.tanh(self.net(x)[..., 0])
