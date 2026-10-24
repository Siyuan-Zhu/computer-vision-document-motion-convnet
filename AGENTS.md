# AGENTS.md

A ConvNet that labels a phone-camera frame of a document as `moving` or `still`, packaged as the `doc_motion` library and CLI.
Library code is in `src/doc_motion`, tests are in `tests/`, and the training settings are in `configs/default.yaml`.

## Gate before every commit

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,tensorflow,jax]"
ruff check . && ruff format --check . && mypy
for b in torch tensorflow jax; do KERAS_BACKEND=$b pytest --cov --cov-fail-under=95 || break; done
```

## Invariants

- The preprocessing in `data.py`, the architecture in `keras_model.py` and the training procedure in `training.py` are a frozen copy of the 2022 method. The README's "Results (2022)" numbers describe exactly this method. Explain odd choices in the README's Design notes and leave them as they are.
- `keras_model.py` builds only from `keras` layers and ops, so the same model runs on the TensorFlow, JAX and PyTorch backends.
- `torch_model.py` mirrors the Keras model layer for layer. After `from_keras`, the parity test in `tests/test_torch_model.py` holds outputs to 1e-5 and the parameter counts equal. Run the gate on all three backends after touching either model file.
- Both backends take float32 NHWC arrays of shape `data.IMAGE_SHAPE`. Labels come from `data.CLASS_NAMES`: `still` = 0, `moving` = 1.
- Every behaviour change gets a test through the public interface (`preprocess`, `load_split`, `training.run`, `predict_folder`, the CLI). Tests build their images from synthetic arrays in `tmp_path`.
- The repo holds code only: images, weights, run folders and CSVs stay out of git, and `.gitignore` already excludes them.
- Describe the data source as "a 2022 applied-ML residency project" and the task as moving versus still document frames. Keep organisation and client names out of code, docs and commit messages.

## Results

- The README reports the 2022 numbers exactly as recorded: 5-fold CV mean 96.2% and held-out test accuracy 0.98 on 597 images. Results from any new run go in their own labelled section.

## Commit

- End commit messages with the `Co-Authored-By` line.
