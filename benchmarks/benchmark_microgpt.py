#!/usr/bin/env python3
"""
Measure ByteDMD cost for a single forward pass of Karpathy's microGPT.
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Uses a tiny configuration (vocab=4, embd=4, heads=2, layers=1, block_size=4).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bytedmd import bytedmd


# --- microGPT components (from Karpathy's microgpt.py) ---

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(v for v in logits)
    exps = [(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    n = len(x)
    ms = sum(xi * xi for xi in x) * (1.0 / n)
    scale = ms ** -0.5
    return [xi * scale for xi in x]


def gpt_forward(wte, wpe, lm_head,
                attn_wq, attn_wk, attn_wv, attn_wo,
                mlp_fc1, mlp_fc2,
                token_id, pos_id, n_head, head_dim):
    """Single-token forward pass of microGPT (1 layer)."""
    tok_emb = wte[token_id]
    pos_emb = wpe[pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # Attention block
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, attn_wq)
    k = linear(x, attn_wk)
    v = linear(x, attn_wv)
    x_attn = []
    for h in range(n_head):
        hs = h * head_dim
        q_h = q[hs:hs+head_dim]
        k_h = k[hs:hs+head_dim]
        v_h = v[hs:hs+head_dim]
        attn_score = sum(q_h[j] * k_h[j] for j in range(head_dim)) / head_dim**0.5
        head_out = v_h
        x_attn.extend(head_out)
    x = linear(x_attn, attn_wo)
    x = [a + b for a, b in zip(x, x_residual)]

    # MLP block
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, mlp_fc1)
    x = [xi * (xi > 0) for xi in x]  # relu
    x = linear(x, mlp_fc2)
    x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, lm_head)
    return logits


def make_matrix(nout, nin):
    return [[1.0] * nin for _ in range(nout)]


# Tiny config
VOCAB_SIZE = 4
N_EMBD = 4
BLOCK_SIZE = 4
N_HEAD = 2
HEAD_DIM = N_EMBD // N_HEAD


if __name__ == '__main__':
    wte = make_matrix(VOCAB_SIZE, N_EMBD)
    wpe = make_matrix(BLOCK_SIZE, N_EMBD)
    lm_head = make_matrix(VOCAB_SIZE, N_EMBD)
    attn_wq = make_matrix(N_EMBD, N_EMBD)
    attn_wk = make_matrix(N_EMBD, N_EMBD)
    attn_wv = make_matrix(N_EMBD, N_EMBD)
    attn_wo = make_matrix(N_EMBD, N_EMBD)
    mlp_fc1 = make_matrix(4 * N_EMBD, N_EMBD)
    mlp_fc2 = make_matrix(N_EMBD, 4 * N_EMBD)

    def forward(wte, wpe, lm_head, attn_wq, attn_wk, attn_wv, attn_wo, mlp_fc1, mlp_fc2):
        return gpt_forward(wte, wpe, lm_head,
                          attn_wq, attn_wk, attn_wv, attn_wo,
                          mlp_fc1, mlp_fc2,
                          token_id=0, pos_id=0, n_head=N_HEAD, head_dim=HEAD_DIM)

    cost = bytedmd(forward, (wte, wpe, lm_head, attn_wq, attn_wk, attn_wv, attn_wo, mlp_fc1, mlp_fc2))
    assert cost == 3214, f"expected 3214, got {cost}"

    print(f"{'Algorithm':<35} {'Operation':<25} {'ByteDMD Cost':>12}")
    print("-" * 75)
    print(f"{'microGPT (1 layer, embd=4)':<35} {'single token forward':<25} {cost:>12}")
    print()
    print(f"Config: vocab={VOCAB_SIZE}, embd={N_EMBD}, heads={N_HEAD}, head_dim={HEAD_DIM}, 1 layer, block_size={BLOCK_SIZE}")
