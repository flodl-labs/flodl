# flodl-hf playground

This directory was scaffolded by `fdl add flodl-hf --playground` inside a flodl project. It's a standalone cargo crate that depends on `flodl-hf` and shows the one-liner `AutoModel` API over BERT / RoBERTa / DistilBERT.

## Getting started

From your project root:

```bash
fdl flodl-hf classify
```

Or directly inside the playground:

```bash
cd flodl-hf && fdl classify
```

The playground takes an optional HuggingFace repo id as its first argument. With none, it loads `cardiffnlp/twitter-roberta-base-sentiment-latest`. Any fine-tuned BERT / RoBERTa / DistilBERT classification checkpoint works — try:

```bash
fdl flodl-hf classify -- nlptown/bert-base-multilingual-uncased-sentiment
fdl flodl-hf classify -- lxyuan/distilbert-base-multilingual-cased-sentiments-student
```

## Feature flavors

`flodl-hf` ships with three profiles, selected via cargo features in `Cargo.toml`:

| Profile     | Features                                      | Use case                                             |
|-------------|-----------------------------------------------|------------------------------------------------------|
| Full        | `default` (`hub` + `tokenizer`)               | Load any model from the Hub, encode text, predict    |
| Vision-only | `default-features = false, features = ["hub"]`| ViT / CLIP towers — no `tokenizers` crate pulled in  |
| Offline     | `default-features = false`                    | `safetensors` loader only, air-gapped pipelines      |

Edit `Cargo.toml`'s `flodl-hf = "=X.Y.Z"` line to pick one. The default (full) is what the playground uses.

## `.bin`-only repos

Some older checkpoints (e.g. `nateraw/bert-base-uncased-emotion`) ship only `pytorch_model.bin` and no `model.safetensors`. Convert once by hand:

```bash
pip install torch transformers safetensors
python - <<'PY'
from transformers import AutoModel
from safetensors.torch import save_file
import os, pathlib
repo_id = "nateraw/bert-base-uncased-emotion"
model = AutoModel.from_pretrained(repo_id)
dest = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface")) / "flodl-converted" / repo_id
dest.mkdir(parents=True, exist_ok=True)
state = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state, dest / "model.safetensors")
print(f"wrote {dest / 'model.safetensors'}")
PY
```

After conversion, `AutoModel::from_pretrained(repo_id)` picks up the local safetensors transparently via the `$HF_HOME/flodl-converted/<repo_id>/` cache. You only need to convert each checkpoint once.

If you prefer the committed script from the flodl repo, grab
[`convert_bin_to_safetensors.py`](https://github.com/flodl-labs/flodl/blob/main/flodl-hf/scripts/convert_bin_to_safetensors.py) and run it directly:

```bash
pip install torch transformers safetensors
python convert_bin_to_safetensors.py nateraw/bert-base-uncased-emotion
```

## Wiring flodl-hf into your main code

This scaffold is a side project for exploration. When you're ready to call flodl-hf from your actual training code, run:

```bash
fdl add flodl-hf --install
```

This appends `flodl-hf = "=X.Y.Z"` to your root `Cargo.toml` `[dependencies]` (default features: `hub` + `tokenizer`).

Example imports:

```rust
use flodl_hf::models::auto::AutoModelForSequenceClassification;
use flodl_hf::models::bert::BertModel;
use flodl_hf::tokenizer::HfTokenizer;
```

## Making this part of a cargo workspace (optional)

By default this playground has its own `target/` dir. To share compilation with your main crate, add a `[workspace]` table to your project's root `Cargo.toml`:

```toml
[workspace]
members = [".", "flodl-hf"]
```

## Docs and next steps

- **Architecture reference**: <https://flodl.dev/guide>
- **`AutoModel` module**: `flodl_hf::models::auto`
- **Per-family modules**: `flodl_hf::models::{bert, roberta, distilbert}` for task-head constructors and custom loading
- **Hub + tokenizer**: `flodl_hf::hub` / `flodl_hf::tokenizer`
