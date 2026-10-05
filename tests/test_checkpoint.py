import torch
from pathlib import Path
from musicgen.models.music_transformer import MusicTransformer
from musicgen.training.optimizer import build_optimizer
from musicgen.training.scheduler import build_scheduler
from musicgen.training.checkpoint import save_checkpoint, load_checkpoint

def cfg():
    return {
        "model":{"architecture":"baseline","d_model":16,"n_layers":1,"n_heads":4,"ffn_dim":32,"dropout":0.0,
                 "max_seq_len":16,"normalization":"rmsnorm","register_embedding":False,"instrument_embedding":False,
                 "num_registers":4,"num_instruments":129},
        "moe":{"enabled":False,"num_experts":3,"routing":"top2","expert_ffn_dim":32,"load_balance_weight":0.01},
        "cross_register":{"enabled":False,"num_heads":4,"dropout":0.0},
        "training":{"learning_rate":1e-3,"weight_decay":0.0,"betas":[0.9,0.95],"max_steps":10},
        "scheduler":{"warmup_steps":1,"min_lr":1e-5},
    }

def test_checkpoint_roundtrip(tmp_path):
    c=cfg()
    m=MusicTransformer(c,50,0)
    opt=build_optimizer(m,c)
    sch=build_scheduler(opt,c)
    x=torch.randint(0,50,(2,8))
    out=m(x, torch.zeros_like(x), torch.full_like(x,128))
    loss=out["logits"].mean()
    loss.backward(); opt.step(); sch.step()
    path=Path(tmp_path)/"x.pt"
    save_checkpoint(path,m,opt,sch,None,2,10,1.2,c)
    m2=MusicTransformer(c,50,0); opt2=build_optimizer(m2,c); sch2=build_scheduler(opt2,c)
    ck=load_checkpoint(path,m2,opt2,sch2,None)
    assert ck["global_step"]==10
