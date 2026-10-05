import math

def perplexity(loss):
    return math.exp(min(float(loss), 20.0))
