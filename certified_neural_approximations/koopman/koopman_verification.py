#!/usr/bin/env python
# coding: utf-8
"""
Trajectory Prediction with Koopman AutoEncoder and ONNX Export for Verification
"""

import sys
import os
import pickle
import torch
import numpy as np
import onnx
import onnxruntime as ort
import shortuuid
from pathlib import Path
from collections import defaultdict
import argparse

from dlkoopman import utils, nets
from dlkoopman.traj_pred import TrajPred, TrajPredDataHandler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(REPO_DIR, "data")


# ---------------------------
# Patched __init__ for ONNX export (removes torch.compile)
# ---------------------------
def _patched_koopman__init__(
    self, dh, encoded_size, encoder_hidden_layers=[100], decoder_hidden_layers=[], batch_norm=False
):
    self.cfg = dh.cfg

    # Define UUID and log file
    self.uuid = shortuuid.uuid()
    '''Modification 1 of 3 wrt to the orginal package.
        Changed the location of log files of dlkoopman to avoid clutter.
    '''
    log_files_dir = "dlkoopman_logs"
    Path(log_files_dir).mkdir(parents=True, exist_ok=True)
    self.log_file = Path(f'./{log_files_dir}/log_{self.uuid}.log').resolve()
    print(f'Log file = {self.log_file}')

    # Get data handler and sizes
    self.dh = dh
    self.input_size = self.dh.Xtr.shape[2]
    self.encoded_size = encoded_size

    # Define AutoEncoder
    self.ae = nets.AutoEncoder(
        input_size=self.input_size,
        encoded_size=self.encoded_size,
        encoder_hidden_layers=encoder_hidden_layers,
        decoder_hidden_layers=decoder_hidden_layers,
        batch_norm=batch_norm
    )
    self.ae.to(dtype=self.cfg.RTYPE, device=self.cfg.DEVICE)
    '''Modification 2 of 3 wrt to the orginal package.
        We just removed the compile function to remove optimizations not supported by the onnx export.
        Just commenented the next two lines.
    '''
    # if utils.is_torch_2() and self.cfg.torch_compile_backend is not None:
    #    self.ae = torch.compile(self.ae, backend=self.cfg.torch_compile_backend)

    # Define linear layer
    self.Knet = nets.Knet(
        size=encoded_size
    )
    self.Knet.to(dtype=self.cfg.RTYPE, device=self.cfg.DEVICE)
    '''Modification 3 of 3 wrt to the orginal package.
        We just removed the compile function to remove optimizations not supported by the onnx export.
        Just commenented the next two lines.
    '''
    # if utils.is_torch_2() and self.cfg.torch_compile_backend is not None:
    #    self.Knet = torch.compile(self.Knet, backend=self.cfg.torch_compile_backend)
    # Define params
    self.params = list(self.ae.parameters()) + list(self.Knet.parameters())

    # Define results
    self.stats = defaultdict(list)

    # Define error flag
    self.error_flag = False

    # Define other attributes to be used later (getter/setters should exist for each of these)
    self.decoder_loss_weight = None
    self.Lambda = None
    self.eigvecs = None


TrajPred.__init__ = _patched_koopman__init__


# ---------------------------
# ONNX Wrapper
# ---------------------------
class TrajPredONNXWrapper(torch.nn.Module):

    """
    We use this wraper to encapsulate all the oprations of the Network described by the package
    for a safe ONNX conversion.
    """

    def __init__(self, ae, knet, num_steps: int, latent_dim: int):
        super().__init__()
        self.encoder = ae.encoder
        self.decoder = ae.decoder
        self.knet = knet
        self.num_steps = num_steps
        self.latent_dim = latent_dim

    def forward(self, x):
        """
        Perform multi-step trajectory prediction in the latent space.
        This is done by emoluting the original architecture forward in the wrapper by:
            - Encoding the Input in the latent space (Encoder)
            - Applying the Koopman Matrix in the latent space (Knet)
            - Decoding the latent space into the Output (Decoder)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim), where input_dim is the observed state dimension.

        Returns:
            torch.Tensor: Reconstructed future trajectory of shape (batch_size, num_steps, input_dim).
        """
        y0 = self.encoder(x)
        y_preds = [y0.unsqueeze(1)]
        y_t = y0
        for _ in range(1, self.num_steps):
            y_t = self.knet(y_t)
            y_preds.append(y_t.unsqueeze(1))
        y_preds = torch.cat(y_preds, dim=1)
        x_preds = self.decoder(y_preds.reshape(-1, self.latent_dim))
        return x_preds.reshape(x.shape[0], self.num_steps, -1)


# ---------------------------
# Export functions
# ---------------------------
def export_onnx_for_testing(tp, onnx_path="trajpred_testing.onnx"):
    """
    Export a TrajPred model to ONNX format with dynamic batch and time dimensions.
    We allow dynamic batching for a much faster comparison with the actual model.

    We cannot use this model for verification beacause, as of now, the verification process
    does not support Shape operations.

    This exported onnx model will be deleted after we assess that the conversion process
    did not introduce any error by comparing it with the orginal model (PyTorch).

    We will export the model also with a fixed batch of 1 in the function 'export_onnx_for_verification'

    Args:
        tp (TrajPred): A trained TrajPred model instance.
        onnx_path (str): Path to save the exported ONNX file.
    """
    model = TrajPredONNXWrapper(tp.ae, tp.Knet, tp.dh.Xtr.shape[1], tp.encoded_size)
    model.eval()
    dummy_input = torch.randn(100, tp.input_size, dtype=tp.cfg.RTYPE, device=tp.cfg.DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["trajectory"],
        dynamic_axes={"input": {0: "batch"}, "trajectory": {0: "batch", 1: "time"}},
        opset_version=17,
        dynamo=False
    )
    print(f"\nPASS: Exported testing ONNX model to: {onnx_path}")


def export_onnx_for_verification(tp, onnx_path="trajpred_ver.onnx"):
    """
    Export a TrajPred model to ONNX format with fixed batch size (for shape-op-safe verification).

    This model will not be tested against the orignal (PyTorch) one beacuse of its fixed size of 1 to
    allow for verification.

    This model will NOT be deleted after execution.

    Args:
        tp (TrajPred): A trained TrajPred model instance.
        onnx_path (str): Path to save the exported ONNX file.
    """
    model = TrajPredONNXWrapper(tp.ae, tp.Knet, tp.dh.Xtr.shape[1], tp.encoded_size)
    model.eval()
    dummy_input = torch.randn(1, tp.input_size, dtype=tp.cfg.RTYPE, device=tp.cfg.DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["trajectory"],
        opset_version=17,
        dynamo=False
    )
    print(f"PASS: Exported verification ONNX model to: {onnx_path}")


# ---------------------------
# Comparison & Inspection
# ---------------------------
def compare_onnx_and_pytorch(tp, onnx_path="trajpred_testing.onnx", num_samples=100, atol=1e-4, rtol=1e-3):
    """
    Compare the prediction outputs from the PyTorch model and the exported ONNX model.

    Args:
        tp (TrajPred): A trained TrajPred model instance.
        onnx_path (str): Path to the ONNX model.
        num_samples (int): Number of random test inputs to generate.
        atol (float): Absolute tolerance for numerical comparison.
        rtol (float): Relative tolerance for numerical comparison.
    """
    X0 = torch.randn(num_samples, tp.input_size, dtype=tp.cfg.RTYPE, device=tp.cfg.DEVICE)
    Xpred_pt = tp.predict_new(X0).cpu().numpy()

    X0_scaled = utils.scale(X0, scale=tp.dh.Xscale) if tp.cfg.normalize_Xdata else X0
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    Xpred_onnx = sess.run(None, {sess.get_inputs()[0].name: X0_scaled.cpu().numpy()})[0]

    if tp.cfg.normalize_Xdata:
        Xpred_onnx = utils.scale(torch.tensor(Xpred_onnx), scale=1/tp.dh.Xscale).numpy()

    Xpred_onnx[:, 0, :] = X0.cpu().numpy()

    diff = np.abs(Xpred_pt - Xpred_onnx)
    print(f"INFO: ONNX vs PyTorch: max_diff={np.max(diff):.6f}, mean_diff={np.mean(diff):.6f}")

    if np.allclose(Xpred_pt, Xpred_onnx, atol=atol, rtol=rtol):
        print("PASS: Outputs match within tolerance")
    else:
        print("FAIL: Outputs differ beyond tolerance")
        sys.exit(1)

    # Clean up test model after comparison
    try:
        os.remove(onnx_path)
        print(f"INFO: Deleted ONNX test model at {onnx_path}")
    except Exception as e:
        print(f"WARNING: Could not delete {onnx_path} - {e}")


def inspect_onnx_model(onnx_path="trajpred.onnx"):
    """
    Compare the prediction outputs from the PyTorch model and the exported ONNX model.

    Args:
        tp (TrajPred): A trained TrajPred model instance.
        onnx_path (str): Path to the ONNX model.
        num_samples (int): Number of random test inputs to generate.
        atol (float): Absolute tolerance for numerical comparison.
        rtol (float): Relative tolerance for numerical comparison.
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    print(f"\nINFO: ONNX Model Info:")
    print(f"- Model IR version: {model.ir_version}")
    print(f"- Model Producer: {model.producer_name} {model.producer_version}")
    print(f"- Opset version: {model.opset_import[0].version}")
    print(f"- Number of nodes (layers): {len(graph.node)}")
    print(f"- Inputs: {[i.name for i in graph.input]}")
    print(f"- Outputs: {[o.name for o in graph.output]}")
    print(f"- Initializers (learned parameters): {len(graph.initializer)}")

    shape_ops = [f"WARNING: Shape Op in node {i}: {node.input}" for i,
                 node in enumerate(graph.node) if node.op_type == "Shape"]
    if shape_ops:
        print("\nWARNING: Detected Shape operations:")
        for msg in shape_ops:
            print(msg)
    else:
        print("PASS: No Shape ops detected.")

    total_params = sum(np.prod(init.dims) for init in graph.initializer)
    print(f"INFO: Total parameters: {total_params}\n")


def print_model_structures(tp, onnx_path):
    """
    Print the structure of the PyTorch and ONNX models.

    Args:
        tp (TrajPred): The trained PyTorch model.
        onnx_path (str): Path to the ONNX model file.
    """
    print("\n--- PyTorch Model Structure ---")
    # Print the wrapper model structure (as used for ONNX export)
    pt_model = TrajPredONNXWrapper(tp.ae, tp.Knet, tp.dh.Xtr.shape[1], tp.encoded_size)
    print(pt_model)

    print("\n--- ONNX Model Structure ---")
    model = onnx.load(onnx_path)
    print(f"ONNX Graph name: {model.graph.name}")
    print("Nodes:")
    for i, node in enumerate(model.graph.node):
        print(f"  {i}: {node.op_type} | Inputs: {node.input} | Outputs: {node.output}")
    print("\nInputs:")
    for inp in model.graph.input:
        print(f"  {inp.name}: {inp.type}")
    print("\nOutputs:")
    for out in model.graph.output:
        print(f"  {out.name}: {out.type}")


# ---------------------------
# Main
# ---------------------------
def train_koopman(
    data, data_path, numepochs=200, encoded_size=64,
    encoder_hidden_layers=[], decoder_hidden_layers=[], batch_norm=False
):
    r"""
    This is an example of usage that works with the provided 'data.pkl' which
    represents a dynamical system with a polynomial slow manifold given as:
        $$\dot{x}_1 = \mu x_1$$
        $$\dot{x}_2 = \lambda\left(x_2-x_1^2\right)$$
    as studied in:
        - the dlkoopman package -- https://github.com/GaloisInc/dlkoopman -- (from which we also adapted part of this documentation),
        - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150171, and
        - https://www.nature.com/articles/s41467-018-07210-0.
    In particular, the data is as used in the first two references.
    The data consists of trajectories, each with 51 indexed states of 2 dimensions.
    Thus, each trajectory is a rollout of an initial state $x_0$ through indexes $\left[x_1,x_2,\cdots,x_{50}\right]$.

    Args:
        data (dict): Dictionary containing the dataset splits:
            - 'Xtr': Training data
            - 'Xva': Validation data
            - 'Xte': Test data
            Each should be a NumPy array of shape (n_samples, time_steps, input_dim).
        data_path (str or Path): Path to the dataset file.
        numepochs (int, optional): Number of epochs for training. Default is 200.
        encoded_size (int, optional): Size of the encoded latent space. Default is 64.
        encoder_hidden_layers (list of int, optional): List specifying the hidden layer sizes of the encoder network. Default is [] (i.e., no hidden layers).
        decoder_hidden_layers (list of int, optional): List specifying the hidden layer sizes of the decoder network. Default is [] (i.e., no hidden layers).
        batch_norm (bool, optional): Whether to use batch normalization in the AutoEncoder. Default is False.
    """

    dataset_name = Path(data_path).stem
    dh = TrajPredDataHandler(Xtr=data['Xtr'], Xva=data['Xva'], Xte=data['Xte'])

    utils.set_seed(10)

    tp = TrajPred(
        dh=dh,
        encoded_size=encoded_size,
        encoder_hidden_layers=encoder_hidden_layers,
        decoder_hidden_layers=decoder_hidden_layers,
        batch_norm=batch_norm
    )
    tp.train_net(
        numepochs=numepochs,
        batch_size=125,
        weight_decay=1e-6
    )

    # In here all the functionalities of the dlkoopman library can be used:
    # For example we could plot the model's stats with 'utils.plot_stats(tp, ['pred_loss', 'total_loss'])'
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    onnx_test_path = os.path.join(DATA_DIR, f"trajpred_{dataset_name}_test.onnx")
    onnx_ver_path = os.path.join(DATA_DIR, f"trajpred_{dataset_name}_ver.onnx")
    torch_path = os.path.join(DATA_DIR, f"trajpred_{dataset_name}.pth")

    export_onnx_for_testing(tp, onnx_test_path)
    compare_onnx_and_pytorch(tp, onnx_test_path)

    export_onnx_for_verification(tp, onnx_ver_path)

    # The optional model inspection can be called as follows:
    inspect_onnx_model(onnx_ver_path)

    # The optional print of the models structures can be called as follows:
    # print_model_structures(tp, onnx_ver_path)

    torch.save(tp, torch_path)
    print(f"PASS: Saved PyTorch model to {torch_path}")
    print(
        f"\nSUCCESS: \n- The model for inference has been successfully created at {torch_path}. \n- A corresponding model for verification has been saved in ONNX format at {onnx_ver_path}.\n")


DEFAULT_FILE = os.path.join(BASE_DIR, "quadratic.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Koopman Trajectory Prediction and ONNX Export")
    parser.add_argument("data-path", nargs="?", default=DEFAULT_FILE, help=f"Path to the data file (default: {DEFAULT_FILE})")
    parser.add_argument("--num-epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--encoded-size", type=int, default=64, help="Encoded latent size (default: 64)")
    parser.add_argument(
        "--encoder-hidden-layers",
        type=int,
        nargs='*',
        default=[],
        help="Encoder hidden layers as a list of ints, e.g. --encoder-hidden-layers 100 50 (default: none)"
    )
    parser.add_argument(
        "--decoder-hidden-layers",
        type=int,
        nargs='*',
        default=[],
        help="Decoder hidden layers as a list of ints, e.g. --decoder-hidden-layers 50 (default: none)"
    )
    parser.add_argument("--batch-norm", action="store_true", help="Use batch normalization (default: False)")

    args = parser.parse_args()

    data_path = args.data_path
    num_epochs = args.num_epochs
    encoded_size = args.encoded_size
    encoder_hidden_layers = args.encoder_hidden_layers
    decoder_hidden_layers = args.decoder_hidden_layers
    batch_norm = args.batch_norm

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_koopman(data, data_path, num_epochs, encoded_size, encoder_hidden_layers, decoder_hidden_layers, batch_norm)
