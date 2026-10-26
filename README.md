# ApproxMoE — Bearing Fault Diagnosis with Mixture-of-Experts

This repository contains the notebook, hardware deployment guide, and benchmark
results for ApproxMoE, the approximate-precision Mixture-of-Experts framework
for bearing fault diagnosis described in the Master's thesis *An Approximate
Precision Mixture-of-Experts Framework for Energy-Efficient Vibration Based
Fault Diagnosis* (Hamzah Farooq, Tampere University, 2026). The system is
evaluated on the CWRU and XJTU-SY datasets and deployed on a Raspberry Pi 4.

## Contents

| File | Description |
|------|-------------|
| `ApproxMoE_Hamzah_Farooq_Thesis.ipynb` | Main Colab notebook: baseline training, difficulty labelling, gate training, quantization-aware training (QAT) of the experts, MoE routing, and evaluation. |
| `hardware_guide_final.md` | Step-by-step guide for exporting the trained models from Colab and deploying and benchmarking them on a Raspberry Pi 4 (latency, memory, CPU, throughput, and energy measurement). |
| `hardware_results.json` | Measured results from the Pi (all models run as FP32): latency, memory, CPU usage, throughput, model sizes, and energy per mode (baseline, expert_A/B/C, moe) on CWRU and XJTU-SY. |

## Important: quantized inference was not run on the Pi

The experts were trained with QAT at 4-bit (Expert A), 8-bit (Expert B), and
INT8-weights/FP32-activations (Expert C). In the software environment used on
the Pi (PyTorch 2.10, 64-bit Raspberry Pi OS), the QAT models could not be
converted into quantized TorchScript modules for the ARM64 CPU. All models
were therefore deployed and measured as **FP32**. The latency and energy
benefits of quantized inference are **not** present in the measured results;
the larger savings reported in the thesis are **analytical projections only**
(thesis Table A.7).

## Summary of results

Measured on the Raspberry Pi 4 (FP32 inference):

- **CWRU (in-domain)**: ApproxMoE reaches 98.9% accuracy (macro F1 0.9873) vs.
  the baseline's 100%.
- **XJTU-SY (cross-domain)**: ApproxMoE reaches 87.9% accuracy (macro F1 0.8734)
  vs. 15.9% for the zero-shot baseline in the software evaluation. The
  Mahalanobis OOD check routes 99.5% of XJTU-SY windows to the cross-domain
  expert without target-domain labels.
- **Latency**: about 75 ms per window for all configurations (about 13.3
  inferences/s, 2.3x faster than real time at 12 kHz). The gate and OOD check
  add 0.9 ms (1.2%).
- **Memory**: 463 MB for the full system.
- **Energy**: differences between configurations are small. Experts B and C use
  6.4% and 7.8% less energy than the baseline on CWRU, and the full system uses
  1.4% more because of routing overhead.

Projected (not measured): with genuine INT8 inference, energy reductions of
56–75% per expert and about 68% for the full system, based on published INT8
speedups on the Cortex-A72. See thesis Section 4.7.4 and Table A.7.

## Getting started

1. Open `ApproxMoE_Hamzah_Farooq_Thesis.ipynb` in Google Colab to see the model
   training and evaluation pipeline.
2. Follow `hardware_guide_final.md` to reproduce the Raspberry Pi deployment
   and hardware benchmarks.
3. See `hardware_results.json` for the measured benchmark data referenced in
   the thesis.

## Hardware setup

- **Device**: Raspberry Pi 4 Model B (4 GB RAM), Raspberry Pi OS Lite 64-bit
- **Software**: Python 3.13, PyTorch 2.10 (CPU), models run as FP32 TorchScript
- **Power measurement**: INA219 current sensor in series with an R&S HMP1042 bench PSU
- **Datasets**: CWRU (896 test windows), XJTU-SY (3,780 windows)
