# ApproxMoE — Bearing Fault Diagnosis with Mixture-of-Experts

This repository contains the notebook, hardware deployment guide, and benchmark
results for a Mixture-of-Experts (MoE) approach to bearing fault diagnosis,
evaluated on CWRU and XJTU-SY datasets and deployed on a Raspberry Pi 4.

## Contents

| File | Description |
|------|-------------|
| `ApproxMoE_Hamzah_Farooq_Thesis.ipynb` | Main Colab notebook — model training, MoE gating, expert routing, and evaluation. |
| `hardware_guide_final.md` | Step-by-step guide for exporting models from Colab and deploying/benchmarking them on a Raspberry Pi 4 (quantization, latency, memory, CPU, throughput, and energy measurement). |
| `hardware_results.json` | Benchmark results collected from the Pi: latency, memory, CPU usage, throughput, model sizes, and energy consumption per mode (baseline, expert_A/B/C, moe) across CWRU and XJTU datasets. |

## Summary of results

The MoE approach routes inputs to lightweight experts based on a learned gate,
aiming to preserve baseline accuracy while cutting latency and energy use.

- **CWRU**: MoE achieves 99.2% accuracy at ~28 ms/inference vs. baseline's 100%
  at ~75 ms/inference — a ~2.7x speedup with minimal accuracy loss.
- **XJTU**: MoE achieves 84.6% accuracy vs. baseline's 49.4%, showing the
  expert routing substantially improves performance on harder, more varied data.
- Full latency, memory, CPU, throughput, and energy breakdowns are in
  `hardware_results.json`; the methodology and Pi setup steps used to produce
  them are documented in `hardware_guide_final.md`.

## Getting started

1. Open `ApproxMoE_Hamzah_Farooq_Thesis.ipynb` in Google Colab to see the model
   training and evaluation pipeline.
2. Follow `hardware_guide_final.md` to reproduce the Raspberry Pi deployment
   and hardware benchmarks.
3. See `hardware_results.json` for the raw benchmark data referenced in the
   thesis.

## Hardware setup

- **Device**: Raspberry Pi 4 (4 GB RAM)
- **Power measurement**: INA219 sensor + R&S HMP1042 bench PSU
- **Datasets**: CWRU (896 test windows), XJTU-SY (3780 test windows)
- **Device**: Raspberry Pi 4 (4 GB RAM)
- **Power measurement**: INA219 sensor + R&S HMP1042 bench PSU
- **Datasets**: CWRU (896 test windows), XJTU-SY (3780 test windows)
