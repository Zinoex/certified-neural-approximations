# Trajectory Prediction with Koopman AutoEncoder and ONNX Export for Verification

This script `koopman_verification.py` demonstrates how to train a Koopman AutoEncoder for trajectory prediction using the [`dlkoopman`](https://github.com/GaloisInc/dlkoopman) package and export the trained model to ONNX format for testing and verification.

---

## Input Format

The script expects a dataset file (typically a `.pkl`) containing a dictionary with:

- `'Xtr'`: Training data
- `'Xva'`: Validation data
- `'Xte'`: Test data

Each should be a NumPy array of shape `(n_samples, time_steps, input_dim)`.

---

## Script Overview

This script includes:

- Custom initialization of `TrajPred` without `torch.compile` (which can cause ONNX export issues).
- Parametric Koopman AutoEncoder model via CLI.
- A wrapper module (`TrajPredONNXWrapper`) for ONNX-friendly inference.
- Export functions:
  - `export_onnx_for_testing`: Dynamic shape export for efficient runtime testing.
  - `export_onnx_for_verification`: Fixed shape export for our verification tool.
- Utility functions to:
  - Compare ONNX and PyTorch model outputs.
  - Inspect ONNX graph structure and detect unsupported operations.
- Saves both ONNX and PyTorch versions of the model using dataset-specific filenames.
---

## Usage

### 1. Prepare Data

Ensure your dataset is saved in a pickle file, e.g., `quadratic.pkl`.

### 2. Run the Script

Basic usage with default model settings:

```bash
python koopman_verification.py ./quadratic.pkl
```

If no file path is provided, it defaults to `./quadratic.pkl`.


Basic usage with default model settings:
```bash
python koopman_verification.py ./quadratic.pkl --num-epochs 200 --encoded-size 64 
```

Arguments (in order). You may omit as many as you like, as long as you provide them in order. The default values are those shown in the example command above:

- data_path: Path to the .pkl file
- numepochs: Number of training epochs (int)
- encoded_size: Size of latent space (int)
- encoder_hidden_layers: Encoder hidden layers (Python list)
- decoder_hidden_layers: Decoder hidden layers (Python list)
- batch_norm: Whether to use batch normalization (True or False)

---

## Output and Logs

- Trains a Koopman AutoEncoder on the dataset for trajectory prediction.
- Saves 
  - PyTorch model: trajpred_\<dataset_name\>.pt
  - ONNX model for verification: trajpred_\<dataset_name\>_ver.onnx
- Prints comparison results (ONNX vs PyTorch)
- Inspects exported ONNX graph (optional)

Example output (with data = "quadratic.pkl" and model inspection but no model structure printing):

```
...
[Training output from the dlkoopman library]
...
PASS: Exported testing ONNX model to: models/trajpred_data_test.onnx
INFO: ONNX vs PyTorch: max_diff=0.000001, mean_diff=0.000000
PASS: Outputs match within tolerance
INFO: Deleted ONNX test model at models/trajpred_data_test.onnx
PASS: Exported verification ONNX model to: models/trajpred_data_ver.onnx

INFO: ONNX Model Info:
- Model IR version: 8
- Model Producer: pytorch 2.7.0
- Opset version: 17
- Number of nodes (layers): 212
- Inputs: ['input']
- Outputs: ['trajectory']
- Initializers (learned parameters): 9
PASS: No Shape ops detected.
INFO: Total parameters: 17562

PASS: Saved PyTorch model to models/trajpred_data.pt

SUCCESS:
- The model for inference has been successfully created at models/trajpred_data.pt.
- A corresponding model for verification has been saved in ONNX format at models/trajpred_data_ver.onnx.
```

---

## Notes on Modifications wrt dlkoopman package

This script makes **two minimal modifications** to the `dlkoopman.TrajPred` class, both in the `__init__` function:

2. Changes the path of the log files to avoid clutter.
2. Removes `torch.compile(...)` to support ONNX export.

These changes do **not** affect model training or internal functionality.

---

## References

- [dlkoopman GitHub](https://github.com/GaloisInc/dlkoopman)
- [dlkoopman on PyPI](https://pypi.org/project/dlkoopman/)

---

## License

MIT License