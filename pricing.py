#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:38:56 2026

@author: aymericlelart
"""
def call_payoff(S, K):
    """
    Calcule le payoff d'un call
    
    S : prix du sous-jacent
    K : strike
    """
    return max(S - K, 0)


def put_payoff(S, K):
    """
    Calcule le payoff d'un put
    """
    return max(K - S, 0)

import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    
    "prix d'un Call avec Black-Scholes "
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    
    "prix d'un Put avec Black-Scholes"
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) 
    
    return put_price
    
    
    