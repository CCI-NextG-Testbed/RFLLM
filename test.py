#!/usr/bin/env python3
from pathlib import Path
import os

import torch

from rfml.data import build_dataset
from rfml.nn.eval import (
    compute_accuracy,
    compute_accuracy_on_cross_sections,
    compute_confusion,
)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener


def main():
    # ------------------------------------------------------------
    # Resolve dataset path robustly (relative to this script file)
    # ------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    dataset_path = (script_dir / "dataset" / "rfml" / "RML2016.10a_dict.pkl").resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file:\n  {dataset_path}\n\n"
            "Fix: put RML2016.10a_dict.pkl at that location, or change dataset_path."
        )

    # ------------------------------------------------------------
    # Build dataset (local path => no network download)
    # ------------------------------------------------------------
    train, val, test, le = build_dataset(
        dataset_name="RML2016.10a",
        path=str(dataset_path),
    )

    # ------------------------------------------------------------
    # Model + trainer (GPU only if available)
    # ------------------------------------------------------------
    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")

    model = build_model(model_name="CNN", input_samples=128, n_classes=len(le))

    trainer = build_trainer(strategy="standard", max_epochs=3, gpu=use_gpu)
    trainer.register_listener(PrintingTrainingListener())
    trainer(model=model, training=train, validation=val, le=le)

    # ------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------
    acc = compute_accuracy(model=model, data=test, le=le)
    acc_vs_snr, snrs = compute_accuracy_on_cross_sections(
        model=model, data=test, le=le, column="SNR"
    )
    cmn = compute_confusion(model=model, data=test, le=le)

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------
    print("===============================")
    print("Overall Testing Accuracy: {:.4f}".format(acc))
    print("SNR (dB)\tAccuracy (%)")
    print("===============================")
    for a, s in zip(acc_vs_snr, snrs):
        # snrs may be numpy ints; cast to int safely
        print("{snr:d}\t{acc:0.1f}".format(snr=int(s), acc=float(a) * 100))
    print("===============================")
    print("Confusion Matrix:")
    print(cmn)

    # ------------------------------------------------------------
    # Save model next to script
    # ------------------------------------------------------------
    out_path = (script_dir / "cnn.pt").resolve()
    model.save(str(out_path))
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()