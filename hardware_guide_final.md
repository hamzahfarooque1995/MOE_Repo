# ApproxMoE Hardware Deployment Guide
## Raspberry Pi 4 — moe@192.168.1.239 (hostname: projectMOE)

---

## Quick reference

| Item | Value |
|------|-------|
| Pi user | moe |
| Pi hostname | projectMOE |
| Pi IP | 192.168.1.239 |
| Pi RAM | 4 GB |
| Pi storage | 512 GB Kingston microSD |
| Python env | moe_env |
| Deployment folder | ~/pi_deployment/ |
| PSU | R&S HMP1042 bench PSU |
| INA219 I2C address | 0x40 |
| Pi idle power | 1.715 W (4.932 V, 350.5 mA) |

---

## Part 1 — Export from Google Colab

Run the `#hardware-export-routing` cell in your notebook. It saves to
`/content/drive/MyDrive/bearing_model/exports/`:

- `baseline_weights.pth`, `expert_A_weights.pth`, `expert_B_weights.pth`, `expert_C_weights.pth`
- `gate_mlp.pt` (TorchScript gate, FP32)
- `gate_scaler.pkl`, `mahal_mu.npy`, `mahal_cov_inv.npy`
- `spec_mean.npy`, `spec_std.npy`, `moe_config.json`
- `model_def.py`, `convert_on_pi.py`

Download the exports folder from Drive as a zip file.

---

## Part 2 — Connect to Pi

Open PowerShell on your Windows laptop:

```
ssh moe@192.168.1.239
```

The virtual environment activates automatically. If not, run:

```
source ~/moe_env/bin/activate
```

---

## Part 3 — Transfer files to the Pi

From PowerShell on your **Windows laptop** (not Pi):

```powershell
scp -r "C:\Users\hamza\Downloads\exports" moe@192.168.1.239:~/pi_deployment
```

Verify on Pi:

```
ls ~/pi_deployment/
```

You should see: `baseline_weights.pth`, `expert_A_weights.pth`, `expert_B_weights.pth`,
`expert_C_weights.pth`, `gate_mlp.pt`, `model_def.py`, `convert_on_pi.py`,
`gate_scaler.pkl`, `mahal_mu.npy`, `mahal_cov_inv.npy`, `spec_mean.npy`, `spec_std.npy`,
`moe_config.json`.

---

## Part 4 — PyTorch ARM quantization issue and fix

**Important:** PyTorch 2.10 deprecated `torch.ao.quantization` eager mode on ARM64.
When you run `convert_on_pi.py` with PyTorch 2.10, all four expert models export
as full FP32 (16.28 MB each) regardless of the QAT configuration applied in Colab.
This means you see no size reduction from quantization on the Pi.

**Fix: downgrade to PyTorch 2.1.2** — the last version where ARM64 eager-mode QAT
conversion works correctly.

```
source ~/moe_env/bin/activate
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```
python3 -c "import torch; print(torch.__version__)"
```

Should print `2.1.2`. Then proceed to Part 5 to convert the models.

---

## Part 5 — Convert models on the Pi

Run the conversion script. This converts QAT weights to quantized TorchScript:

```
python3 ~/pi_deployment/convert_on_pi.py
```

Then copy converted models to the deployment folder:

```
cp ~/pi_deployment/exports/*.pt ~/pi_deployment/ 2>/dev/null || true
```

Verify sizes. With PyTorch 2.1.2 you should see:

| Model | Expected size |
|-------|--------------|
| baseline.pt | 16.28 MB |
| expert_A.pt | ~2.0 MB |
| expert_B.pt | ~4.1 MB |
| expert_C.pt | ~4.1 MB |
| gate_mlp.pt | ~0.02 MB |

Check:

```
ls -lh ~/pi_deployment/*.pt
```

---

## Part 6 — Transfer real test datasets to Pi

The benchmark scripts use the actual CWRU and XJTU test windows, not synthetic data.
Export from Colab and transfer to the Pi.

**In Colab**, add this cell to export test windows:

```python
#hardware-export-datasets
import numpy as np, os

EXPORT_DIR = '/content/drive/MyDrive/bearing_model/exports'
os.makedirs(EXPORT_DIR, exist_ok=True)

# CWRU test set (896 windows)
np.save(os.path.join(EXPORT_DIR, 'cwru_test_windows.npy'), X_test_spec)
np.save(os.path.join(EXPORT_DIR, 'cwru_test_labels.npy'),  y_test)

# XJTU-SY evaluation set
np.save(os.path.join(EXPORT_DIR, 'xjtu_windows.npy'), X_xjtu)
np.save(os.path.join(EXPORT_DIR, 'xjtu_labels.npy'),  y_xjtu)

for fname in ['cwru_test_windows.npy','cwru_test_labels.npy',
              'xjtu_windows.npy','xjtu_labels.npy']:
    path = os.path.join(EXPORT_DIR, fname)
    arr  = np.load(path)
    print(f"{fname}: {arr.shape}")
print("Dataset export complete.")
```

Download and transfer to Pi:

```powershell
scp "C:\Users\hamza\Downloads\cwru_test_windows.npy" moe@192.168.1.239:~/pi_deployment/
scp "C:\Users\hamza\Downloads\cwru_test_labels.npy"  moe@192.168.1.239:~/pi_deployment/
scp "C:\Users\hamza\Downloads\xjtu_windows.npy"      moe@192.168.1.239:~/pi_deployment/
scp "C:\Users\hamza\Downloads\xjtu_labels.npy"       moe@192.168.1.239:~/pi_deployment/
```

Verify on Pi:

```
ls -lh ~/pi_deployment/*.npy
python3 -c "
import numpy as np
cwru_w = np.load('/root/pi_deployment/cwru_test_windows.npy')
xjtu_w = np.load('/root/pi_deployment/xjtu_windows.npy')
print('CWRU windows:', cwru_w.shape)
print('XJTU windows:', xjtu_w.shape)
"
```

Expected: CWRU windows `(896, 128, 128)`, XJTU windows `(3780, 128, 128)`.

---

## Part 7 — The inference engine

Create `moe_inference.py` on the Pi:

```
cat > ~/moe_inference.py << 'PYEOF'
#!/usr/bin/env python3
import os, json, time, joblib
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.quantized.engine = 'qnnpack'

DEPLOY_DIR = os.path.expanduser('~/pi_deployment')

with open(os.path.join(DEPLOY_DIR, 'moe_config.json')) as f:
    cfg = json.load(f)

WINDOW_SIZE   = cfg['window_size']
SPEC_SIZE     = cfg['spec_size']
N_FFT         = cfg['n_fft']
HOP_LENGTH    = cfg['hop_length']
OOD_THRESHOLD = cfg['ood_threshold']
ESC_THRESHOLD = cfg['escalation_threshold']
SAMPLE_RATE   = cfg['cwru_fs']
NUM_CLASSES   = cfg['num_classes']
CLASS_NAMES   = {int(k): v for k, v in cfg['class_names'].items()}

print("Loading models...")
t0 = time.perf_counter()
models = {}
for name, fname in [('baseline','baseline.pt'),('expert_A','expert_A.pt'),
                    ('expert_B','expert_B.pt'),('expert_C','expert_C.pt'),
                    ('gate','gate_mlp.pt')]:
    path = os.path.join(DEPLOY_DIR, fname)
    models[name] = torch.jit.load(path, map_location='cpu')
    models[name].eval()
    print(f"  {name:<12} {os.path.getsize(path)/1024/1024:.2f} MB")
print(f"Loaded in {time.perf_counter()-t0:.2f}s")

gate_scaler   = joblib.load(os.path.join(DEPLOY_DIR, 'gate_scaler.pkl'))
mahal_mu      = np.load(os.path.join(DEPLOY_DIR, 'mahal_mu.npy'))
mahal_cov_inv = np.load(os.path.join(DEPLOY_DIR, 'mahal_cov_inv.npy'))
spec_mean     = np.load(os.path.join(DEPLOY_DIR, 'spec_mean.npy'))
spec_std      = np.load(os.path.join(DEPLOY_DIR, 'spec_std.npy'))
print("Artifacts loaded.")

def predict_from_spectrogram(spec_128x128, mode='moe'):
    """
    spec_128x128: numpy array (128, 128) — pre-computed normalised spectrogram
    Returns dict with prediction, confidence, expert_used, timings.
    """
    timings  = {}
    t_total  = time.perf_counter()

    # Extract scalar features for routing
    # Features are approximated from spectrogram for gate input
    t0   = time.perf_counter()
    spec = spec_128x128.astype(np.float64)
    rms  = np.sqrt(np.mean(spec**2))
    mu   = spec.mean(); sigma = spec.std() + 1e-8
    kurt = np.mean(((spec - mu) / sigma) ** 4)
    crest   = np.max(np.abs(spec)) / (rms + 1e-8)
    skew    = np.mean(((spec - mu) / sigma) ** 3)
    low_e   = np.mean(spec[:spec.shape[0]//4, :] ** 2)
    high_e  = np.mean(spec[spec.shape[0]//2:, :] ** 2)
    flat    = spec.flatten()
    pn      = flat - flat.min(); pn = pn / (pn.sum() + 1e-8)
    entropy = -np.sum(pn * np.log(pn + 1e-12))
    feats   = np.array([rms, kurt, crest, skew, low_e, high_e, entropy], dtype=np.float32)
    feats_sc = gate_scaler.transform([feats])[0].astype(np.float32)
    timings['feature_ms'] = (time.perf_counter() - t0) * 1000

    # Routing decision
    t0 = time.perf_counter()
    expert_used = mode
    if mode == 'moe':
        d = feats_sc - mahal_mu
        dist = float(np.sqrt(max(d @ mahal_cov_inv @ d, 0.0)))
        if dist > OOD_THRESHOLD:
            expert_used = 'expert_C'
        else:
            feat_t = torch.tensor(feats_sc).unsqueeze(0)
            with torch.no_grad():
                tier = torch.argmax(torch.softmax(models['gate'](feat_t), dim=1)).item()
            expert_used = 'expert_A' if tier == 0 else 'expert_B'
    timings['routing_ms'] = (time.perf_counter() - t0) * 1000

    # Inference
    t0  = time.perf_counter()
    key = expert_used if expert_used in models else 'expert_C'
    x   = torch.tensor(spec_128x128, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = models[key](x)
        pred   = torch.argmax(logits, dim=1).item()
        conf   = torch.softmax(logits, dim=1).max().item()

    if expert_used == 'expert_A' and conf < ESC_THRESHOLD:
        with torch.no_grad():
            logits = models['expert_B'](x)
            pred   = torch.argmax(logits, dim=1).item()
            conf   = torch.softmax(logits, dim=1).max().item()
        expert_used = 'expert_B (escalated)'

    timings['inference_ms'] = (time.perf_counter() - t0) * 1000
    timings['total_ms']     = (time.perf_counter() - t_total) * 1000

    return {
        'prediction':  pred,
        'class_name':  CLASS_NAMES[pred],
        'confidence':  round(conf, 4),
        'expert_used': expert_used,
        'timings':     {k: round(v, 3) for k, v in timings.items()},
    }

if __name__ == '__main__':
    print("\n── Quick test on CWRU windows ─────────────────")
    CWRU_W = os.path.join(DEPLOY_DIR, 'cwru_test_windows.npy')
    if os.path.exists(CWRU_W):
        windows = np.load(CWRU_W)
        labels  = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_labels.npy'))
        for mode in ['baseline', 'moe']:
            r = predict_from_spectrogram(windows[0], mode=mode)
            print(f"  [{mode:<10}] {r['class_name']:<14} conf={r['confidence']:.3f}"
                  f"  via {r['expert_used']:<22} {r['timings']['total_ms']:.1f} ms")
    else:
        print("  CWRU windows not found. Transfer datasets first (Part 6).")
    print("\nInference engine ready.")
PYEOF
```

Test it:

```
python3 ~/moe_inference.py
```

---

## Part 8 — Latency benchmark (real datasets)

```
cat > ~/benchmark_latency.py << 'PYEOF'
#!/usr/bin/env python3
"""
Latency benchmark using real CWRU and XJTU test spectrograms.
"""
import os, json
import numpy as np
from moe_inference import predict_from_spectrogram, DEPLOY_DIR

N_WARMUP = 20
N_RUNS   = 100

CWRU_W = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_windows.npy'))
CWRU_L = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_labels.npy'))
XJTU_W = np.load(os.path.join(DEPLOY_DIR, 'xjtu_windows.npy'))
XJTU_L = np.load(os.path.join(DEPLOY_DIR, 'xjtu_labels.npy'))

print(f"CWRU: {CWRU_W.shape}  XJTU: {XJTU_W.shape}")

def benchmark(windows, labels, mode, dataset_name):
    print(f"  {mode} on {dataset_name} ({N_RUNS} runs)...")
    for i in range(N_WARMUP):
        predict_from_spectrogram(windows[i % len(windows)], mode=mode)

    times, preds, routing = [], [], {}
    for i in range(N_RUNS):
        r = predict_from_spectrogram(windows[i % len(windows)], mode=mode)
        times.append(r['timings']['total_ms'])
        preds.append(r['prediction'])
        e = r['expert_used']
        routing[e] = routing.get(e, 0) + 1

    # accuracy against true labels
    true = [int(labels[i % len(labels)]) for i in range(N_RUNS)]
    acc  = sum(p == t for p, t in zip(preds, true)) / N_RUNS * 100

    return {
        'mode':            mode,
        'dataset':         dataset_name,
        'n_runs':          N_RUNS,
        'accuracy_pct':    round(acc, 2),
        'latency_mean_ms': round(np.mean(times), 3),
        'latency_std_ms':  round(np.std(times), 3),
        'latency_p50_ms':  round(np.percentile(times, 50), 3),
        'latency_p95_ms':  round(np.percentile(times, 95), 3),
        'latency_p99_ms':  round(np.percentile(times, 99), 3),
        'expert_routing':  routing,
    }

results = []
for mode in ['baseline', 'expert_A', 'expert_B', 'expert_C', 'moe']:
    results.append(benchmark(CWRU_W, CWRU_L, mode, 'CWRU'))
for mode in ['baseline', 'expert_C', 'moe']:
    results.append(benchmark(XJTU_W, XJTU_L, mode, 'XJTU'))

with open(os.path.expanduser('~/benchmark_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'Mode':<20} {'Dataset':<6} {'Acc%':>6} {'Mean ms':>9} {'P50 ms':>8} {'P95 ms':>8}")
print("-" * 63)
for r in results:
    print(f"{r['mode']:<20} {r['dataset']:<6} {r['accuracy_pct']:>6.1f}"
          f" {r['latency_mean_ms']:>9.2f} {r['latency_p50_ms']:>8.2f} {r['latency_p95_ms']:>8.2f}")
print("\nSaved: ~/benchmark_results.json")
PYEOF
python3 ~/benchmark_latency.py
```

---

## Part 9 — Memory usage

```
cat > ~/measure_memory.py << 'PYEOF'
#!/usr/bin/env python3
import os, json
import numpy as np
from moe_inference import predict_from_spectrogram, DEPLOY_DIR

windows = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_windows.npy'))

def get_ram_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if 'VmRSS' in line:
                return int(line.split()[1]) / 1024
    return 0.0

results = []
print(f"{'Mode':<20} {'RAM (MB)':>10}")
print("-" * 32)
for mode in ['baseline', 'expert_A', 'expert_B', 'expert_C', 'moe']:
    for i in range(20):
        predict_from_spectrogram(windows[i % len(windows)], mode=mode)
    ram = get_ram_mb()
    print(f"{mode:<20} {ram:>10.1f}")
    results.append({'mode': mode, 'ram_mb': round(ram, 1)})

with open(os.path.expanduser('~/memory_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: ~/memory_results.json")
PYEOF
python3 ~/measure_memory.py
```

---

## Part 10 — CPU usage

```
cat > ~/measure_cpu.py << 'PYEOF'
#!/usr/bin/env python3
import os, json, time, threading
import numpy as np
from moe_inference import predict_from_spectrogram, DEPLOY_DIR

windows = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_windows.npy'))

def get_cpu():
    with open('/proc/stat') as f:
        parts = f.readline().split()
    idle  = int(parts[4])
    total = sum(int(x) for x in parts[1:])
    return idle, total

def measure(mode, n=100):
    readings = []
    stop = threading.Event()
    def reader():
        pi, pt = get_cpu()
        while not stop.is_set():
            time.sleep(0.1)
            ci, ct = get_cpu()
            if ct - pt > 0:
                readings.append(100 * (1 - (ci - pi) / (ct - pt)))
            pi, pt = ci, ct
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    for i in range(n):
        predict_from_spectrogram(windows[i % len(windows)], mode=mode)
    stop.set(); th.join()
    return round(np.mean(readings), 1) if readings else 0

results = []
print(f"{'Mode':<20} {'CPU %':>8}")
print("-" * 30)
for mode in ['baseline', 'expert_A', 'expert_B', 'expert_C', 'moe']:
    cpu = measure(mode)
    print(f"{mode:<20} {cpu:>8.1f} %")
    results.append({'mode': mode, 'cpu_pct': cpu})

with open(os.path.expanduser('~/cpu_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: ~/cpu_results.json")
PYEOF
python3 ~/measure_cpu.py
```

---

## Part 11 — Throughput

```
cat > ~/measure_throughput.py << 'PYEOF'
#!/usr/bin/env python3
import os, json, time
import numpy as np
from moe_inference import predict_from_spectrogram, DEPLOY_DIR

cwru_w = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_windows.npy'))
xjtu_w = np.load(os.path.join(DEPLOY_DIR, 'xjtu_windows.npy'))

N = 200

results = []
print(f"{'Mode':<20} {'Dataset':<6} {'inf/s':>10} {'ms/inf':>10}")
print("-" * 50)

for windows, dataset_name, modes in [
    (cwru_w, 'CWRU', ['baseline','expert_A','expert_B','expert_C','moe']),
    (xjtu_w, 'XJTU', ['baseline','expert_C','moe'])]:
    for mode in modes:
        for i in range(20):
            predict_from_spectrogram(windows[i % len(windows)], mode=mode)
        t0 = time.perf_counter()
        for i in range(N):
            predict_from_spectrogram(windows[i % len(windows)], mode=mode)
        elapsed = time.perf_counter() - t0
        ips = round(N / elapsed, 2)
        ms  = round(elapsed * 1000 / N, 3)
        print(f"{mode:<20} {dataset_name:<6} {ips:>10.2f} {ms:>10.2f}")
        results.append({'mode': mode, 'dataset': dataset_name,
                        'inferences_per_sec': ips, 'ms_per_sample': ms})

with open(os.path.expanduser('~/throughput_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: ~/throughput_results.json")
PYEOF
python3 ~/measure_throughput.py
```

---

## Part 12 — Model sizes

```
cat > ~/measure_model_sizes.py << 'PYEOF'
#!/usr/bin/env python3
import os, json

DEPLOY_DIR = os.path.expanduser('~/pi_deployment')
models = {
    'baseline': 'baseline.pt',
    'expert_A': 'expert_A.pt',
    'expert_B': 'expert_B.pt',
    'expert_C': 'expert_C.pt',
    'gate_mlp': 'gate_mlp.pt',
}
results = []
print(f"{'Model':<20} {'Size (MB)':>10}")
print("-" * 32)
for name, fname in models.items():
    path = os.path.join(DEPLOY_DIR, fname)
    if os.path.exists(path):
        size_mb = round(os.path.getsize(path) / 1024 / 1024, 3)
        print(f"{name:<20} {size_mb:>10.3f} MB")
        results.append({'model': name, 'size_mb': size_mb})

with open(os.path.expanduser('~/model_size_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: ~/model_size_results.json")
PYEOF
python3 ~/measure_model_sizes.py
```

---

## Part 13 — INA219 energy measurement (R&S HMP1042 PSU)

### 13.1 Wiring

The R&S HMP1042 bench PSU feeds the INA219, which feeds the Pi GPIO 5V pin.
The INA219 sits in series with the 5V supply line so it can measure current.

```
HMP1042 positive  →  INA219 VIN+
INA219 VIN-       →  Pi GPIO Pin 2  (5V)
HMP1042 negative  →  Pi GPIO Pin 6  (Ground)

INA219 VCC        →  Pi GPIO Pin 1  (3.3V)
INA219 SDA        →  Pi GPIO Pin 3  (GPIO2 SDA1)
INA219 SCL        →  Pi GPIO Pin 5  (GPIO3 SCL1)
```

Set the HMP1042 output to **5.0 V**, current limit 2.0 A before powering on.

### 13.2 Enable I2C on the Pi

```
sudo raspi-config nonint do_i2c 0
sudo reboot
```

Reconnect after reboot:

```
ssh moe@192.168.1.239
```

### 13.3 Install dependencies

```
sudo apt update
sudo apt install -y i2c-tools swig python3-dev liblgpio-dev
source ~/moe_env/bin/activate
pip install adafruit-circuitpython-ina219 adafruit-blinka
```

### 13.4 Verify detection

```
sudo i2cdetect -y 1
```

You should see `40` in the grid at row 40, column 0.

### 13.5 Quick readout test

```
python3 -c "
import board
from adafruit_ina219 import INA219
ina = INA219(board.I2C())
print(f'Voltage: {ina.bus_voltage:.3f} V')
print(f'Current: {ina.current:.2f} mA')
print(f'Power:   {ina.power:.4f} W')
"
```

Expected at idle: ~4.932 V, ~350.5 mA, ~1.715 W.

### 13.6 Energy measurement script

```
cat > ~/measure_energy.py << 'PYEOF'
#!/usr/bin/env python3
"""
Measures energy per inference using INA219 and R&S HMP1042 PSU.
Runs each mode on both CWRU and XJTU real test windows.
"""
import os, json, time, threading
import numpy as np
from moe_inference import predict_from_spectrogram, DEPLOY_DIR

try:
    import board
    from adafruit_ina219 import INA219
    ina219 = INA219(board.I2C())
    HAS_INA219 = True
    print("INA219 connected.")
    print(f"  Bus voltage: {ina219.bus_voltage:.3f} V")
    print(f"  Current    : {ina219.current:.2f} mA")
    print(f"  Power      : {ina219.power:.4f} W")
except Exception as e:
    print(f"INA219 not available ({e}) — latency only.")
    HAS_INA219 = False

CWRU_W = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_windows.npy'))
CWRU_L = np.load(os.path.join(DEPLOY_DIR, 'cwru_test_labels.npy'))
XJTU_W = np.load(os.path.join(DEPLOY_DIR, 'xjtu_windows.npy'))
XJTU_L = np.load(os.path.join(DEPLOY_DIR, 'xjtu_labels.npy'))

def read_ina():
    if not HAS_INA219:
        return 0.0, 0.0, 0.0
    try:
        return ina219.bus_voltage, ina219.current, ina219.power
    except:
        return 0.0, 0.0, 0.0

def measure_idle(duration_s=10):
    print(f"\nMeasuring idle ({duration_s}s)...")
    v_l, i_l, p_l = [], [], []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        v, i, p = read_ina()
        v_l.append(v); i_l.append(i); p_l.append(p)
        time.sleep(0.1)
    r = {
        'idle_voltage_v':  round(float(np.mean(v_l)), 4),
        'idle_current_ma': round(float(np.mean(i_l)), 4),
        'idle_power_w':    round(float(np.mean(p_l)), 6),
    }
    print(f"  Idle: {r['idle_power_w']:.4f} W  "
          f"({r['idle_current_ma']:.1f} mA @ {r['idle_voltage_v']:.3f} V)")
    return r

def measure_mode(windows, dataset_name, mode, n=200):
    print(f"\n  {mode} on {dataset_name} ({n} inferences)...")
    for i in range(20):
        predict_from_spectrogram(windows[i % len(windows)], mode=mode)

    v_l, i_l, p_l = [], [], []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            v, i, p = read_ina()
            v_l.append(v); i_l.append(i); p_l.append(p)
            time.sleep(0.1)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t0 = time.perf_counter()
    for i in range(n):
        predict_from_spectrogram(windows[i % len(windows)], mode=mode)
    elapsed = time.perf_counter() - t0
    stop.set(); th.join()

    ms   = elapsed * 1000 / n
    mp   = float(np.mean(p_l)) if p_l else 0.0
    mv   = float(np.mean(v_l)) if v_l else 0.0
    mi   = float(np.mean(i_l)) if i_l else 0.0
    e_mj = mp * ms

    r = {
        'mode':              mode,
        'dataset':           dataset_name,
        'n_inferences':      n,
        'ms_per_inference':  round(ms, 3),
        'mean_voltage_v':    round(mv, 4),
        'mean_current_ma':   round(mi, 4),
        'mean_power_w':      round(mp, 6),
        'energy_per_inf_mj': round(e_mj, 6),
        'ina219_available':  HAS_INA219,
    }
    print(f"    Latency: {ms:.2f} ms/inf  Power: {mp:.4f} W  Energy: {e_mj:.4f} mJ")
    return r

if __name__ == '__main__':
    all_results = {}
    all_results['idle'] = measure_idle(duration_s=10)

    mode_results = []
    print("\nCWRU dataset:")
    for mode in ['baseline', 'expert_A', 'expert_B', 'expert_C', 'moe']:
        r = measure_mode(CWRU_W, 'CWRU', mode, n=200)
        if HAS_INA219:
            r['active_power_w']   = round(r['mean_power_w'] - all_results['idle']['idle_power_w'], 6)
            r['active_energy_mj'] = round(r['active_power_w'] * r['ms_per_inference'], 6)
        mode_results.append(r)

    print("\nXJTU dataset:")
    for mode in ['baseline', 'expert_C', 'moe']:
        r = measure_mode(XJTU_W, 'XJTU', mode, n=200)
        if HAS_INA219:
            r['active_power_w']   = round(r['mean_power_w'] - all_results['idle']['idle_power_w'], 6)
            r['active_energy_mj'] = round(r['active_power_w'] * r['ms_per_inference'], 6)
        mode_results.append(r)

    all_results['modes'] = mode_results

    out = os.path.expanduser('~/energy_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    print(f"\n{'Mode':<20} {'Dataset':<6} {'ms/inf':>8} {'Power W':>9} {'Active W':>9} {'Energy mJ':>11}")
    print("-" * 68)
    for r in mode_results:
        p = r.get('mean_power_w', 0)
        a = r.get('active_power_w', 0)
        e = r.get('active_energy_mj', r.get('energy_per_inf_mj', 0))
        print(f"{r['mode']:<20} {r['dataset']:<6} {r['ms_per_inference']:>8.2f}"
              f" {p:>9.4f} {a:>9.4f} {e:>11.4f}")
PYEOF
python3 ~/measure_energy.py
```

---

## Part 14 — Collect all results

```
cat > ~/collect_all_results.py << 'PYEOF'
#!/usr/bin/env python3
import os, json

files = {
    'latency':     '~/benchmark_results.json',
    'memory':      '~/memory_results.json',
    'cpu':         '~/cpu_results.json',
    'throughput':  '~/throughput_results.json',
    'model_sizes': '~/model_size_results.json',
    'energy':      '~/energy_results.json',
}

all_results = {}
for key, path in files.items():
    path = os.path.expanduser(path)
    if os.path.exists(path):
        with open(path) as f:
            all_results[key] = json.load(f)
        print(f"  Loaded: {key}")
    else:
        print(f"  Missing (skip): {path}")

out = os.path.expanduser('~/all_hardware_results.json')
with open(out, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nAll results: {out}")
PYEOF
python3 ~/collect_all_results.py
```

---

## Part 15 — Copy results to laptop

Open PowerShell on your **Windows laptop**:

```powershell
scp moe@192.168.1.239:~/all_hardware_results.json "C:\Users\hamza\Downloads\"
scp moe@192.168.1.239:~/benchmark_results.json    "C:\Users\hamza\Downloads\"
scp moe@192.168.1.239:~/energy_results.json       "C:\Users\hamza\Downloads\"
scp moe@192.168.1.239:~/memory_results.json       "C:\Users\hamza\Downloads\"
scp moe@192.168.1.239:~/cpu_results.json          "C:\Users\hamza\Downloads\"
scp moe@192.168.1.239:~/throughput_results.json   "C:\Users\hamza\Downloads\"
scp moe@192.168.1.239:~/model_size_results.json   "C:\Users\hamza\Downloads\"
```

---

## What to report in the thesis

| Measurement | Script | Output file |
|-------------|--------|------------|
| Latency (mean, P50, P95, P99) | `benchmark_latency.py` | `benchmark_results.json` |
| RAM usage | `measure_memory.py` | `memory_results.json` |
| CPU usage | `measure_cpu.py` | `cpu_results.json` |
| Throughput (inf/s) | `measure_throughput.py` | `throughput_results.json` |
| Model size on disk | `measure_model_sizes.py` | `model_size_results.json` |
| Energy per inference | `measure_energy.py` | `energy_results.json` |

All six roll up into `all_hardware_results.json` via `collect_all_results.py`.

### Key metrics to highlight

- **Model size reduction**: Expert A (4-bit) vs Baseline (FP32) — target ~8x reduction if Fix 1 applied
- **Energy per inference**: Active power × latency for each configuration
- **MoE routing overhead**: MoE mean latency vs Baseline mean latency
- **Accuracy preservation**: MoE accuracy vs Baseline accuracy on same test windows

### Actual results obtained (FP32, before Fix 1)

| Mode | Dataset | Mean ms | Acc % | Active W | Energy mJ |
|------|---------|---------|-------|----------|-----------|
| baseline | CWRU | 76.16 | 100.0 | 0.549 | 41.8 |
| expert_A | CWRU | 76.51 | 100.0 | 0.521 | 39.8 |
| expert_B | CWRU | 76.31 | 100.0 | 0.399 | 30.5 |
| expert_C | CWRU | 76.30 | 61.8 | 0.368 | 28.1 |
| moe | CWRU | 76.74 | 99.2 | 0.562 | 43.1 |
| baseline | XJTU | 75.60 | 49.4 | 0.357 | 27.0 |
| expert_C | XJTU | 75.69 | 85.0 | 0.446 | 33.7 |
| moe | XJTU | 75.68 | 84.6 | 0.376 | 28.5 |

These are FP32 baseline numbers. After applying Fix 1 (PyTorch 2.1.2 downgrade)
and rerunning the conversion, Expert A should drop to ~2 MB and ~22 ms/inf,
giving roughly 75% energy reduction on the Easy routing path.

