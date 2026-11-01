"""
Modal launch file for the DSA4212 small-LLM project.

The GPU is chosen here, in code (gpu="L4"), not on the Modal website.
Matches DECLARATIONS.md §1: 1 x NVIDIA L4, bf16, standard (preemptible) pricing.

Usage (from the repo root, after `pip install modal` and `modal setup`):
    modal run modal_app.py::gpu_check          # ~1 min, confirms you got an L4
    modal run modal_app.py::train --config configs/baseline.yaml --seed 0
"""
import subprocess
import modal

# ---------------------------------------------------------------------------
# Container image: the software every run uses (pin exact versions later in
# requirements.txt, as DECLARATIONS.md §1 requires).
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tokenizers", "pyyaml", "datasets")
    .add_local_dir("code", remote_path="/root/code")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Persistent storage for the prepared data and run outputs.
# Each group member's account gets its own copy (same name, same contents).
data_vol = modal.Volume.from_name("small-llm-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("small-llm-runs", create_if_missing=True)

app = modal.App("dsa4212-small-llm", image=image)

GPU = "L4"  # <-- the declared hardware. Do not change for final runs.


@app.function(gpu=GPU, timeout=10 * 60)
def gpu_check():
    """Quick check: prints the GPU name and confirms bf16 works. Costs ~1 cent."""
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    import torch
    name = torch.cuda.get_device_name(0)
    assert "L4" in name, f"Expected an L4, got {name}"
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    print(f"OK: {name}, bf16 supported = {torch.cuda.is_bf16_supported()}")


@app.function(
    gpu=GPU,
    # Modal stops a function after 5 minutes by default. A run needs 60 min of
    # counted training plus paused validation/checkpointing, so allow 2 hours.
    timeout=2 * 60 * 60,
    # If the container is preempted or crashes, Modal restarts it; train.py
    # resumes from the last checkpoint with its counted time carried over.
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def train(config: str, seed: int = 0):
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert "L4" in gpu_name, f"Wrong GPU: {gpu_name}"  # DECLARATIONS.md §1
    print(f"GPU: {gpu_name}")

    # train.py (our modified nanoGPT) is not written yet; this is where it plugs in.
    subprocess.run(
        ["python", "/root/code/train.py", f"--config=/root/{config}",
         f"--seed={seed}", "--data_dir=/data", "--out_root=/runs",
         f"--gpu_name={gpu_name}"],
        check=True,
    )
    runs_vol.commit()  # make sure results are saved to the Volume


@app.function(cpu=8, memory=32768, timeout=4 * 60 * 60,
              volumes={"/data": data_vol})
def prepare_data():
    """CPU-only (cheap): download, split, train tokenizer, tokenize. Phase 0."""
    subprocess.run(["python", "/root/code/prepare_data.py", "--out=/data"], check=True)
    data_vol.commit()
