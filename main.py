#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Pricing & Greeks Visualisation - Black-Scholes Model
Author: Aymeric Lelart

This script generates two figures:
  1. Payoff vs Black-Scholes price (Call & Put)
  2. Greeks across a range of underlying prices (Delta, Gamma, Vega, Theta, Rho)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pricing import (
    call_payoff, put_payoff,
    black_scholes_call, black_scholes_put,
    delta, gamma, vega, theta, rho
)

# ──────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────

K     = 100    # Strike price
T     = 1      # Time to maturity (1 year)
r     = 0.05   # Risk-free rate (5%)
sigma = 0.20   # Volatility (20%)

S_values = list(range(50, 151))  # Underlying price range: 50 to 150


# ──────────────────────────────────────────
# COMPUTE PRICES & PAYOFFS
# ──────────────────────────────────────────

call_payoffs  = [call_payoff(S, K) for S in S_values]
put_payoffs   = [put_payoff(S, K)  for S in S_values]
call_prices   = [black_scholes_call(S, K, T, r, sigma) for S in S_values]
put_prices    = [black_scholes_put(S, K, T, r, sigma)  for S in S_values]


# ──────────────────────────────────────────
# COMPUTE GREEKS
# ──────────────────────────────────────────

call_delta  = [delta(S, K, T, r, sigma, "call") for S in S_values]
put_delta   = [delta(S, K, T, r, sigma, "put")  for S in S_values]
gammas      = [gamma(S, K, T, r, sigma)          for S in S_values]
vegas       = [vega(S, K, T, r, sigma)            for S in S_values]
call_theta  = [theta(S, K, T, r, sigma, "call") for S in S_values]
put_theta   = [theta(S, K, T, r, sigma, "put")  for S in S_values]
call_rho    = [rho(S, K, T, r, sigma, "call")   for S in S_values]
put_rho     = [rho(S, K, T, r, sigma, "put")    for S in S_values]


# ──────────────────────────────────────────
# FIGURE 1 — PAYOFF VS BLACK-SCHOLES PRICE
# ──────────────────────────────────────────

fig1, ax = plt.subplots(figsize=(10, 5))

ax.plot(S_values, call_payoffs, 'b--',  label="Call Payoff (at expiry)")
ax.plot(S_values, put_payoffs,  'r--',  label="Put Payoff (at expiry)")
ax.plot(S_values, call_prices,  'b-',   label="Call Price — Black-Scholes", linewidth=2)
ax.plot(S_values, put_prices,   'r-',   label="Put Price — Black-Scholes",  linewidth=2)
ax.axvline(x=K, color='gray', linestyle=':', label=f"Strike K = {K}")

ax.set_xlabel("Underlying Price (S)")
ax.set_ylabel("Option Value ($)")
ax.set_title("Payoff vs Black-Scholes Price — European Options", fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figure1_pricing.png", dpi=150)
plt.show()
print("Figure 1 saved: figure1_pricing.png")


# ──────────────────────────────────────────
# FIGURE 2 — GREEKS DASHBOARD
# ──────────────────────────────────────────

fig2 = plt.figure(figsize=(14, 9))
fig2.suptitle("Greeks Dashboard — Black-Scholes Model\n"
              f"K={K}, T={T}y, r={r:.0%}, σ={sigma:.0%}",
              fontweight='bold', fontsize=13)

gs = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35)

# --- Delta ---
ax1 = fig2.add_subplot(gs[0, 0])
ax1.plot(S_values, call_delta, 'b-', label="Call Delta")
ax1.plot(S_values, put_delta,  'r-', label="Put Delta")
ax1.axvline(x=K, color='gray', linestyle=':')
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_title("Delta\n(price sensitivity to S)", fontsize=10)
ax1.set_xlabel("S"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# --- Gamma ---
ax2 = fig2.add_subplot(gs[0, 1])
ax2.plot(S_values, gammas, 'g-', label="Gamma (call = put)")
ax2.axvline(x=K, color='gray', linestyle=':')
ax2.set_title("Gamma\n(rate of change of Delta)", fontsize=10)
ax2.set_xlabel("S"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# --- Vega ---
ax3 = fig2.add_subplot(gs[0, 2])
ax3.plot(S_values, vegas, 'm-', label="Vega (call = put)")
ax3.axvline(x=K, color='gray', linestyle=':')
ax3.set_title("Vega\n(sensitivity to volatility, per 1%)", fontsize=10)
ax3.set_xlabel("S"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# --- Theta ---
ax4 = fig2.add_subplot(gs[1, 0])
ax4.plot(S_values, call_theta, 'b-', label="Call Theta")
ax4.plot(S_values, put_theta,  'r-', label="Put Theta")
ax4.axvline(x=K, color='gray', linestyle=':')
ax4.axhline(y=0, color='black', linewidth=0.5)
ax4.set_title("Theta\n(time decay, per day)", fontsize=10)
ax4.set_xlabel("S"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# --- Rho ---
ax5 = fig2.add_subplot(gs[1, 1])
ax5.plot(S_values, call_rho, 'b-', label="Call Rho")
ax5.plot(S_values, put_rho,  'r-', label="Put Rho")
ax5.axvline(x=K, color='gray', linestyle=':')
ax5.axhline(y=0, color='black', linewidth=0.5)
ax5.set_title("Rho\n(sensitivity to interest rate, per 1%)", fontsize=10)
ax5.set_xlabel("S"); ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

# --- Summary table ---
ax6 = fig2.add_subplot(gs[1, 2])
ax6.axis('off')
S_atm = K  # At-the-money summary
summary = [
    ["Greek",  "Call (ATM)",  "Put (ATM)"],
    ["Delta",  f"{delta(S_atm,K,T,r,sigma,'call'):.3f}",
               f"{delta(S_atm,K,T,r,sigma,'put'):.3f}"],
    ["Gamma",  f"{gamma(S_atm,K,T,r,sigma):.4f}", "same"],
    ["Vega",   f"{vega(S_atm,K,T,r,sigma):.4f}",  "same"],
    ["Theta",  f"{theta(S_atm,K,T,r,sigma,'call'):.4f}",
               f"{theta(S_atm,K,T,r,sigma,'put'):.4f}"],
    ["Rho",    f"{rho(S_atm,K,T,r,sigma,'call'):.4f}",
               f"{rho(S_atm,K,T,r,sigma,'put'):.4f}"],
]
tbl = ax6.table(cellText=summary[1:], colLabels=summary[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)
ax6.set_title("ATM Greeks Summary\n(S = K = 100)", fontsize=10)

plt.savefig("figure2_greeks.png", dpi=150)
plt.show()
print("Figure 2 saved: figure2_greeks.png")

