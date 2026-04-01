#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:47:03 2026

@author: aymericlelart


payoff d'un Call'
"""
from pricing import call_payoff, put_payoff, black_scholes_call, black_scholes_put
import matplotlib.pyplot as plt

# ---------------
# PARAMÈTRES ----
# ---------------
K = 100
T = 1
r = 0.05
sigma= 0.2




# ---------------------
# Valeurs De S --------
# ---------------------

S_values = list(range(50, 151)) # nombre de 50 à 150 




# ---------------------
# calcul --------------
# ---------------------

call_values = []
put_values = []


for S in S_values:
    call_values.append(call_payoff(S, K))
    put_values.append(put_payoff(S, K)) 
    




# ---------------------
# PRIX BLACK-SCHOLES --
# ---------------------

call_price = []

for S in S_values :
    call_price.append(black_scholes_call(S, K, T, r, sigma))
    

put_price = []
for S in S_values : 
    put_price.append(black_scholes_put(S, K, T, r, sigma))
    
    
    
    
# ---------------------
# Graphique -----------
# ---------------------


plt.plot(S_values, call_values, label= "Call Payoff")
plt.plot(S_values, put_values, label= "Put Payoff")
plt.plot(S_values, call_price, label= "Call Price (BS)")
plt.plot(S_values, put_price, label= "Put Price (BS)")

plt.xlabel("Prix du sous-Jacent (S)")
plt.ylabel("Valeurs")
plt.title("Payoff VS Black-Scholes price")
plt.axvline(x=K, linestyle='--', label="Strike")

plt.legend()
plt.grid()

plt.show()
    

