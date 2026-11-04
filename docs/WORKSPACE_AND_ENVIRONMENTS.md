# Workspace, dependencies, and execution environment

## Repository layout and ownership

`AuditRepair-v8.3/` is the frozen main/corrected study; `AuditRepair-v8.4/` contains E5, E6, E8 and E7 gold-blind tasks; `e7_collected*/`, `e7_rc2/`, `results/`, `code/` form the separately qualified E7 result add-on; `historical_support/` is a frozen earlier source/results archive, not the active evaluator. The working copy lives **outside** these sealed folders under `reproductions/` or another untracked directory.

## Environment facts

- Supported original reproduction target: Ubuntu 24.04 / WSL, Python 3.12.
- Canonical pinned CPU package set: `AuditRepair-v8.3/requirements.txt` (NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8 and transitive packages).
- Exact wheel-hash lock is not provided; Python/BLAS patch differences can change low-order floating scores. Reproduction checks compare scientific decisions and also record numeric differences, rather than relying on array byte identity alone.
- E1–E6/E8 and E7-from-cache run on CPU; E7 fresh annotation collection requires a CUDA GPU with sufficient capacity for each pinned model, a compatible vLLM container and Hugging Face access as appropriate.
- Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` for reproducibility.
- Recorded model revision/container evidence is under `e7_collected/provenance`, `e7_collected_rc2/provenance` and `AuditRepair-v8.4/E7_tasks/`.

## WSL Quickstart (PowerShell then Ubuntu)

```powershell
wsl -d Ubuntu-24.04
```

```bash
cd /mnt/c/AuditRepair  # adjust to your clone path
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv "$HOME/.venvs/auditrepair-release"
source "$HOME/.venvs/auditrepair-release/bin/activate"
python -m pip install -r AuditRepair-v8.3/requirements.txt
bash VERIFY_COMPLETE.sh --quick
```

Run the longer reproduction scripts from `REPRODUCE.md`; do not store vLLM caches or downloaded model weights in the Git checkout.
