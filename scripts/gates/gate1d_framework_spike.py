"""Gate 1d - can Keras 3 actually build the model we designed?

Decision 2 in the plan picks Keras 3 on TensorFlow because the course labs are TF/Keras.
That is a good reason, but it was made without building anything. The real model is five
heterogeneous inputs fused by masked multi-head attention over a *variable-length set of
edits*, and ragged/masked attention is precisely where the Keras functional API tends to
fight back. Finding that out in week 3 of a four-week project is fatal.

So: build the awkward shape now, on random data, and check three things.

  1. it compiles and trains an epoch
  2. **masking actually masks** - padded rows must not change the output
  3. the code is legible enough that WS3 will still want to work in it next week

Test 2 is the one that matters. A model that silently attends to padding trains fine,
scores fine, and is wrong - and the failure looks like "geometry did not help".

Pass -> Keras 3 confirmed. Fail -> PyTorch, and update plan sections 10 and 15.

Run in the 3.11 venv; TensorFlow does not support the machine's 3.14:

    .venv311/Scripts/python.exe scripts/gate1d_framework_spike.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

RESULTS = Path(__file__).resolve().parents[2] / "results" / "gates"

# Shapes mirror the real design (PROJECT_PLAN section 5.3), scaled down so the spike
# runs on a CPU in under a minute.
N = 96            # changesets
K = 24            # max edits per changeset (real value 256)
RASTER = 16       # geometry raster side (real value larger)
CHARS = 64        # char-CNN sequence length
VOCAB = 70
USER_STEPS = 20   # user history length (real value 200)
USER_FEATS = 4
TAB = 12
EDIT_FEATS = 8


def build_model(keras, layers):
    """Five branches, attention pooling over a masked set of edits, one sigmoid out."""
    # B1 geometry: 2-channel before/after raster, one per sampled edit
    geom_in = keras.Input(shape=(K, RASTER, RASTER, 2), name="geometry")
    g = layers.TimeDistributed(layers.Conv2D(8, 3, activation="relu", padding="same"))(geom_in)
    g = layers.TimeDistributed(layers.MaxPooling2D(2))(g)
    g = layers.TimeDistributed(layers.GlobalAveragePooling2D())(g)      # (N, K, 8)

    # B2 text: char-CNN with multi-width parallel kernels (the Inception idea)
    text_in = keras.Input(shape=(CHARS,), dtype="int32", name="text")
    t = layers.Embedding(VOCAB, 16, mask_zero=False)(text_in)
    t = layers.Concatenate()([
        layers.GlobalMaxPooling1D()(layers.Conv1D(12, w, activation="relu", padding="same")(t))
        for w in (3, 5, 7)
    ])

    # per-edit tabular features, concatenated onto the geometry embedding
    edit_in = keras.Input(shape=(K, EDIT_FEATS), name="edit_features")
    mask_in = keras.Input(shape=(K,), name="edit_mask")        # 1 = real edit, 0 = padding
    e = layers.Concatenate(axis=-1)([g, edit_in])
    e = layers.TimeDistributed(layers.Dense(32, activation="relu"))(e)

    # B3 edit sequence: BiLSTM over the same set, in commit order
    seq = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(
        e, mask=keras.ops.cast(mask_in, "bool")
    )

    # attention pooling over the K edits. The mask is the whole point of this gate.
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)
    pooled = attn(seq, seq, attention_mask=_pair_mask(keras, mask_in))
    pooled = layers.GlobalAveragePooling1D()(pooled, mask=keras.ops.cast(mask_in, "bool"))

    # B4 user history GRU
    user_in = keras.Input(shape=(USER_STEPS, USER_FEATS), name="user_sequence")
    u = layers.GRU(16)(user_in)

    # B5 tabular MLP
    tab_in = keras.Input(shape=(TAB,), name="tabular")
    b5 = layers.Dense(24, activation="relu")(tab_in)

    fused = layers.Concatenate()([pooled, t, u, b5])
    h = layers.Dense(32, activation="relu")(fused)
    h = layers.Dropout(0.2)(h)
    h = layers.LayerNormalization()(h)
    out = layers.Dense(1, activation="sigmoid", name="p_damaging")(h)

    return keras.Model([geom_in, text_in, edit_in, mask_in, user_in, tab_in], out)


def _pair_mask(keras, mask_in):
    """(N, K) key mask -> (N, K, K) attention mask MultiHeadAttention expects."""
    m = keras.ops.cast(mask_in, "bool")
    return keras.ops.logical_and(m[:, :, None], m[:, None, :])


def fake_batch(rng, n=N, real_edits=None):
    real = rng.integers(1, K + 1, size=n) if real_edits is None else np.full(n, real_edits)
    mask = (np.arange(K)[None, :] < real[:, None]).astype("float32")
    return {
        "geometry": rng.normal(size=(n, K, RASTER, RASTER, 2)).astype("float32") * mask[:, :, None, None, None],
        "text": rng.integers(0, VOCAB, size=(n, CHARS)).astype("int32"),
        "edit_features": (rng.normal(size=(n, K, EDIT_FEATS)) * mask[:, :, None]).astype("float32"),
        "edit_mask": mask,
        "user_sequence": rng.normal(size=(n, USER_STEPS, USER_FEATS)).astype("float32"),
        "tabular": rng.normal(size=(n, TAB)).astype("float32"),
    }, mask, real


def main() -> int:
    started = time.monotonic()
    result = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate": "1d",
        "question": "does Keras 3 handle five inputs plus masked attention over a variable-length edit set?",
    }

    try:
        import keras
        from keras import layers
        import tensorflow as tf  # noqa: F401  - backend
        result["keras_version"] = keras.__version__
        result["backend"] = keras.backend.backend()
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = "CANNOT RUN"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["note"] = "TensorFlow needs Python <= 3.13. Use .venv311 or run on Colab."
        _write(result)
        return 2

    rng = np.random.default_rng(20260904)

    try:
        model = build_model(keras, layers)
        result["params"] = int(model.count_params())
        result["built"] = True
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = "FAIL"
        result["stage"] = "build"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(result)
        return 1

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    try:
        x, mask, real = fake_batch(rng)
        y = (rng.random(N) > 0.5).astype("float32")
        hist = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
        result["trained"] = True
        result["loss"] = float(hist.history["loss"][0])
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = "FAIL"
        result["stage"] = "train"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(result)
        return 1

    # --- the test that matters -------------------------------------------------
    # Take a batch, note the predictions, then scribble garbage into the PADDED slots
    # only. If masking works the predictions are unchanged. If it does not, the model
    # has been attending to padding all along and every result would be quietly wrong.
    x2, mask2, _ = fake_batch(rng, n=32, real_edits=8)
    p_before = model.predict(x2, verbose=0)

    pad = (1.0 - mask2)[:, :, None]
    x3 = {k: v.copy() for k, v in x2.items()}
    x3["edit_features"] = x2["edit_features"] + pad * rng.normal(size=x2["edit_features"].shape) * 50
    x3["geometry"] = x2["geometry"] + pad[:, :, :, None, None] * rng.normal(size=x2["geometry"].shape) * 50
    p_after = model.predict(x3, verbose=0)

    drift = float(np.max(np.abs(p_before - p_after)))
    result["padding_perturbation_max_delta"] = drift
    result["masking_holds"] = bool(drift < 1e-4)

    result["seconds"] = round(time.monotonic() - started, 1)
    result["verdict"] = "PASS" if result["masking_holds"] else "FAIL - masking leaks into the output"
    result["decision"] = (
        "Keras 3 confirmed (plan Decision 2)"
        if result["masking_holds"]
        else "switch to PyTorch and update plan sections 10 and 15"
    )
    _write(result)
    return 0 if result["masking_holds"] else 1


def _write(result: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "gate1d_framework_spike.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
