import math


def binomial_price(S, K, r, sigma, T, steps, is_call, american=False, q=0.0):
    if steps <= 0 or S <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return float('nan')

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        return float('nan')

    # Terminal payoffs: V[j] = node with j up-moves
    V = [0.0] * (steps + 1)
    for j in range(steps + 1):
        ST = S * (u ** j) * (d ** (steps - j))
        V[j] = max(ST - K, 0.0) if is_call else max(K - ST, 0.0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * V[j + 1] + (1.0 - p) * V[j])
            if american:
                Sij = S * (u ** j) * (d ** (i - j))
                ex = (Sij - K) if is_call else (K - Sij)
                if ex < 0.0:
                    ex = 0.0
                V[j] = max(cont, ex)
            else:
                V[j] = cont
    return V[0]