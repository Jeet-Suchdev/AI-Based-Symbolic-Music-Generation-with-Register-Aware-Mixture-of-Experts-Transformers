from musicgen.data.dataset import TinySyntheticDataset, collate_batch

def test_tiny_dataset():
    ds=TinySyntheticDataset(vocab_size=100,seq_len=16,n=4)
    b=collate_batch([ds[0],ds[1]],pad_id=0)
    assert b["input_ids"].shape[0]==2
    assert b["labels"].shape==b["input_ids"].shape
    assert b["register_ids"].shape==b["input_ids"].shape
