# Register-Aware Mixture-of-Experts Transformer for Symbolic Music Generation

A research-grade PyTorch project for autoregressive symbolic MIDI generation using a decoder-only Transformer with:

- REMI symbolic tokenization through MidiTok
- register embeddings (LOW / MID / HIGH)
- learned top-1/top-2 Mixture-of-Experts routing
- cross-register information flow
- load-balancing loss
- mixed precision
- gradient accumulation
- validation/perplexity
- atomic periodic/best/latest checkpoints
- full optimizer/scheduler/scaler/RNG resume
- MIDI generation
- evaluation scaffolding
- unit tests
- Google Colab launcher

No LSTM, GRU, or RNN is used.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows activation:

```powershell
.venv\Scripts\activate
```

## 2. Put MIDI files in

```text
data/raw/
```

The preprocessing script recursively finds `.mid` and `.midi` files.

## 3. Preprocess

```bash
python scripts/preprocess.py --input data/raw --output data/processed
```

## 4. Build tokenized shards

```bash
python scripts/build_dataset.py \
  --input data/processed \
  --output data/tokenized \
  --config configs/cross_register_moe.yaml
```

This creates train/validation/test shards and saves the tokenizer.

## 5. Run tests

```bash
pytest -q
```

## 6. Tiny smoke training

Use:

```bash
python scripts/train.py \
  --config configs/cross_register_moe.yaml \
  --tiny-overfit
```

This uses a tiny deterministic synthetic fallback dataset if no tokenized data is available, allowing the model/checkpoint pipeline to be validated before spending GPU time.

## 7. Train

```bash
python scripts/train.py \
  --config configs/cross_register_moe.yaml
```

## 8. Resume

Automatic resume:

```bash
python scripts/train.py \
  --config configs/cross_register_moe.yaml \
  --resume
```

Explicit checkpoint:

```bash
python scripts/train.py \
  --config configs/cross_register_moe.yaml \
  --resume checkpoints/step_10000.pt
```

The checkpoint stores model, optimizer, scheduler, AMP scaler, epoch, global step, best validation loss, configuration and RNG states.

## 9. Generate MIDI

```bash
python scripts/generate.py \
  --checkpoint checkpoints/best.pt \
  --output generated/sample.mid \
  --max-tokens 1024 \
  --temperature 0.9 \
  --top-k 50 \
  --top-p 0.95
```

## 10. Colab

Open `notebooks/colab_training.ipynb`.

The notebook:

1. clones this repository if a GitHub URL is supplied;
2. installs dependencies;
3. mounts Google Drive;
4. checks the NVIDIA GPU;
5. optionally copies a dataset from Drive;
6. trains with the same Python scripts;
7. stores checkpoints on Drive;
8. resumes after a disconnected runtime.

For a large dataset, prefer storing tokenized shards on fast local Colab storage or cloud/object storage rather than repeatedly reading thousands of small files from Drive.

## Architecture

```text
MIDI
  |
  v
validation / metadata
  |
  v
MidiTok REMI
  |
  v
token IDs + register IDs
  |
  v
token + register + instrument embeddings
  |
  v
decoder-only Transformer
  |
  v
learned MoE router
  |---------|---------|
 LOW      MID       HIGH
  |---------|---------|
       cross-register fusion
              |
              v
       vocabulary projection
              |
              v
       next-token prediction
              |
              v
             MIDI
```

## Experimental configurations

- `transformer_baseline.yaml`
- `register_transformer.yaml`
- `moe_transformer.yaml`
- `cross_register_moe.yaml`

Do not claim MoE improves music quality until the ablations demonstrate it.

## Dataset leakage

Splits are made by MIDI file/composition before sequence-window construction. Never randomly split windows from the same composition across train and validation/test.

## Recommended workflow

1. Run tests.
2. Run tiny overfit.
3. Train on a small real subset.
4. Verify checkpoint/resume.
5. Generate MIDI.
6. Scale the dataset.
7. Run baseline/register/MoE/cross-register ablations.

## Fully automated Google Colab pipeline

The project now supports an end-to-end Colab run with no manual MIDI collection. The default dataset is **MAESTRO v3.0.0 MIDI-only**, downloaded from the official Magenta dataset source. The official page reports 1,276 performances and about 7.04 million notes across the 198.7-hour dataset; the MIDI-only archive is about 56 MB compressed. MAESTRO v3.0.0 is released under CC BY-NC-SA 4.0, so check the license before using it beyond research/education. The official split is retained by the builder when metadata is present.

### Colab flow

```text
Fresh Colab
  -> clone repository
  -> install dependencies
  -> mount Google Drive
  -> download MAESTRO automatically + verify SHA256
  -> extract/cache MIDI
  -> validate and remove corrupt/invalid/duplicate files
  -> preserve official train/validation/test split
  -> tokenize with REMI
  -> create fixed-length shards
  -> train Cross-Register MoE Transformer
  -> save latest/best/periodic checkpoints to Drive
  -> automatically resume after disconnect
  -> generate MIDI
  -> evaluate generated MIDI
```

Run `notebooks/colab_training.ipynb`. The main command is:

```bash
python scripts/run_pipeline.py \
  --config configs/cross_register_moe.yaml \
  --dataset maestro \
  --drive-root /content/drive/MyDrive/musicgen_checkpoints
```

The pipeline is idempotent: existing downloads, cleaned data, tokenized shards, and checkpoints are reused. Use `--force-data` only when you deliberately want to rebuild the data.

### Dataset note

MAESTRO is excellent for a controlled first research experiment because it is high-quality symbolic piano data, but it is **not a diverse multi-instrument dataset**. For final claims about general multi-instrument music generation, add a second appropriately licensed multi-instrument MIDI corpus to the dataset registry rather than presenting MAESTRO alone as representative of all music.
