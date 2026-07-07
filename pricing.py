#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Pricing Module - Black-Scholes Model
Author: Aymeric Lelart

Provides:
- Call and Put payoff functions (at expiry)
- Black-Scholes pricing for Call and Put
- Greeks: Delta, Gamma, Vega, Theta, Rho
"""

import numpy as np
from scipy.stats import norm


# ──────────────────────────────────────────
# PAYOFF FUNCTIONS (at expiry)
# ──────────────────────────────────────────

def call_payoff(S, K):
    """
    Payoff of a European Call option at expiry.

    Parameters:
        S (float): Underlying asset price at expiry
        K (float): Strike price

    Returns:
        float: max(S - K, 0)
    """
    return max(S - K, 0)


def put_payoff(S, K):
    """
    Payoff of a European Put option at expiry.

    Parameters:
        S (float): Underlying asset price at expiry
        K (float): Strike price

    Returns:
        float: max(K - S, 0)
    """
    return max(K - S, 0)


# ──────────────────────────────────────────
# BLACK-SCHOLES HELPER
# ──────────────────────────────────────────

def _d1_d2(S, K, T, r, sigma):
    """
    Compute d1 and d2 used in the Black-Scholes formula.

    Parameters:
        S     (float): Current underlying price
        K     (float): Strike price
        T     (float): Time to maturity (in years)
        r     (float): Risk-free interest rate (annualised)
        sigma (float): Volatility of the underlying (annualised)

    Returns:
        tuple: (d1, d2)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# ──────────────────────────────────────────
# BLACK-SCHOLES PRICING
# ──────────────────────────────────────────

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes price of a European Call option.

    Parameters:
        S     (float): Current underlying price
        K     (float): Strike price
        T     (float): Time to maturity (in years)
        r     (float): Risk-free interest rate (annualised)
        sigma (float): Volatility of the underlying (annualised)

    Returns:
        float: Call option price
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Black-Scholes price of a European Put option.

    Parameters:
        S     (float): Current underlying price
        K     (float): Strike price
        T     (float): Time to maturity (in years)
        r     (float): Risk-free interest rate (annualised)
        sigma (float): Volatility of the underlying (annualised)

    Returns:
        float: Put option price
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ──────────────────────────────────────────
# GREEKS
# ──────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type="call"):
    """
    Delta — sensitivity of option price to changes in the underlying price.
    Interpretation: how much the option price moves for a $1 move in S.

    Parameters:
        option_type (str): "call" or "put"

    Returns:
        float: Delta (between 0 and 1 for calls, -1 and 0 for puts)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """
    Gamma — rate of change of Delta with respect to the underlying price.
    Same value for calls and puts.
    Interpretation: measures convexity; high gamma = Delta changes quickly.

    Returns:
        float: Gamma (always positive)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Vega — sensitivity of option price to changes in volatility.
    Same value for calls and puts.
    Interpretation: how much the option price changes for a 1% move in sigma.

    Returns:
        float: Vega (expressed per 1% change in volatility)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01  # scaled to 1% vol move


def theta(S, K, T, r, sigma, option_type="call"):
    """
    Theta — time decay; sensitivity of option price to passage of time.
    Interpretation: how much value the option loses per calendar day.

    Parameters:
        option_type (str): "call" or "put"

    Returns:
        float: Theta per day (typically negative)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def rho(S, K, T, r, sigma, option_type="call"):
    """
    Rho — sensitivity of option price to changes in the risk-free rate.
    Interpretation: how much the option price changes for a 1% move in r.

    Parameters:
        option_type (str): "call" or "put"

    Returns:
        float: Rho (expressed per 1% change in interest rate)
    """
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
    
    
