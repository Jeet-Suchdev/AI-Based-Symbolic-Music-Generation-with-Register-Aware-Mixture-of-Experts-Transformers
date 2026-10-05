import torch
from musicgen.models.music_transformer import MusicTransformer

def cfg():
    return {
        "model": {"architecture":"cross_register_moe","d_model":32,"n_layers":2,"n_heads":4,"ffn_dim":64,
                  "dropout":0.0,"max_seq_len":32,"normalization":"rmsnorm","register_embedding":True,
                  "instrument_embedding":True,"num_registers":4,"num_instruments":129},
        "moe":{"enabled":True,"num_experts":3,"routing":"top2","expert_ffn_dim":64,"load_balance_weight":0.01},
        "cross_register":{"enabled":True,"num_heads":4,"dropout":0.0}
    }

def test_forward():
    m = MusicTransformer(cfg(), 100, 0)
    x = torch.randint(0,100,(2,16))
    r = torch.randint(0,4,(2,16))
    i = torch.randint(0,129,(2,16))
    y = m(x,r,i)
    assert y["logits"].shape == (2,16,100)
