"""The paper's math: Black-Scholes greeks, hedging strategies, paths, P&L and the entropic price.

Units follow the paper: spot starts at 1, strike K = STRIKE[derivative], risk aversion 1 on absolute P&L.
Greeks return (price/K, delta, gamma*K), so everything is scale-free in log-moneyness s = log(S/K).
"""

import math

import torch

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
