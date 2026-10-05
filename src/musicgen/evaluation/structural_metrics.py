from collections import Counter

def token_repetition_ratio(ids, n=4):
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    counts = Counter(grams)
    repeated = sum(c-1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))
