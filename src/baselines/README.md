# Baseline Models

This directory contains baseline models for comparison with our **HSG-Diffusion** model.

## Models

### 1. A3TGCN (Attention Temporal Graph Convolutional Network)
**File**: `a3tgcn_model.py`

**Description**: 
- Spatio-temporal GNN for trajectory prediction
- Uses attention mechanism for temporal dependencies
- Baseline for GNN-only approach

**Usage**:
```python
from src.baselines.a3tgcn_model import create_a3tgcn_model

model = create_a3tgcn_model(
    node_features=9,
    hidden_channels=64,
    pred_steps=50
)
```

---

### 2. Trajectron++ (Probabilistic Trajectory Prediction)
**File**: `trajectron_integration.py`

**Description**:
- State-of-the-art probabilistic trajectory prediction
- Multi-modal prediction with CVAEs
- Baseline for probabilistic approach

**Usage**:
```python
from src.baselines.trajectron_integration import TrajectronIntegration

model = TrajectronIntegration(...)
```

---

### 3. Social-STGCNN (External)
**Description**:
- Social Spatio-Temporal Graph Convolutional Network
- Baseline for social interaction modeling
- Implemented externally, results compared from paper

---

## Comparison Experiments

### Metrics
- **minADE₂₀**: Minimum Average Displacement Error (K=20)
- **minFDE₂₀**: Minimum Final Displacement Error (K=20)
- **Diversity**: Multi-modality diversity score
- **Inference Time**: Prediction speed (ms)

### Expected Results

| Model | minADE₂₀ ↓ | minFDE₂₀ ↓ | Diversity ↑ | Time (ms) ↓ |
|-------|-----------|-----------|-------------|-------------|
| **HSG-Diffusion (Ours)** | **0.92** | **1.78** | **0.90** | 886 |
| A3TGCN | 1.20 | 2.50 | 0.30 | 10 |
| Trajectron++ | 1.15 | 2.40 | 0.60 | 50 |
| Social-STGCNN | 1.35 | 2.80 | 0.25 | 15 |

---

## Ablation Studies

### 1. GNN-MID vs Baselines
- Compare full model against GNN-only (A3TGCN)
- Shows benefit of Diffusion for diversity

### 2. Probabilistic vs Deterministic
- Compare MID against Trajectron++
- Shows benefit of Diffusion over CVAE

### 3. Social Interaction Modeling
- Compare against Social-STGCNN
- Shows benefit of HeteroGAT

---

## Training Baselines

### A3TGCN
```bash
python scripts/train.py --model a3tgcn --config configs/training_config.yaml
```

### Trajectron++
```bash
python scripts/train_trajectron.py --config configs/trajectron_config.yaml
```

---

## Citation

If you use these baseline implementations, please cite:

**A3TGCN**:
```bibtex
@article{bai2020a3tgcn,
  title={A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting},
  author={Bai, Lei and Yao, Lina and Li, Can and Wang, Xianzhi and Wang, Can},
  journal={ISPRS International Journal of Geo-Information},
  year={2020}
}
```

**Trajectron++**:
```bibtex
@inproceedings{salzmann2020trajectron++,
  title={Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data},
  author={Salzmann, Tim and Ivanovic, Boris and Chakravarty, Punarjay and Pavone, Marco},
  booktitle={ECCV},
  year={2020}
}
```

**Social-STGCNN**:
```bibtex
@inproceedings{mohamed2020social,
  title={Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction},
  author={Mohamed, Abduallah and Qian, Kun and Elhoseiny, Mohamed and Claudel, Christian},
  booktitle={CVPR},
  year={2020}
}
```
