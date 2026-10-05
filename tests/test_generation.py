def test_import_generation():
    from musicgen.generation.sampler import sample_next_token
    import torch
    x=sample_next_token(torch.randn(1,20), temperature=1.0, top_k=5, top_p=0.9)
    assert x.shape==(1,1)
