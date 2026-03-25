import torch 

class KVCache:
    def __init__(self, n_layers: int, device=None):
        self.k_cache, self.v_cache = [], []
        for _ in range(n_layers):
            self.k_cache.append(torch.tensor([], device=device))
            self.v_cache.append(torch.tensor([], device=device))

    def update(self, k, v, i):
        if self.k_cache[i].numel() == 0:
            self.k_cache[i] = k
            self.v_cache[i] = v
        else:
            self.k_cache[i] = torch.cat([self.k_cache[i], k], dim=-2)
            self.v_cache[i] = torch.cat([self.v_cache[i], v], dim=-2)
        return self.k_cache[i], self.v_cache[i]

    def get_seq_length(self, i = 0):
        i = 2
        if len(self.k_cache) <= i or self.k_cache[i].numel() == 0:
            return 0
        return self.k_cache[i].shape[-2]