# Stage 1 Frozen Baselines

Status: complete

Output directory: `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines`

| Baseline | Train PR-AUC | Val PR-AUC | Val Rank@1 | Val MRR | Val ECE | Calibration |
|---|---:|---:|---:|---:|---:|---|
| B0 | 0.248639 | 0.249819 | 0.275100 | 0.539250 | 0.019647 | train_positive_rate |
| B0k | 0.285498 | 0.284894 | 0.215863 | 0.474337 | 0.015764 | train_section_kind_prior |
| B1 | 0.722833 | 0.701831 | 0.782129 | 0.877644 | 0.240610 | validation_temperature |
| B2 | 0.273169 | 0.277541 | 0.334337 | 0.588438 | 0.037303 | validation_platt |
| B2b | 0.289548 | 0.286501 | 0.322289 | 0.573186 | 0.021303 | validation_platt |
| B3 | 0.711528 | 0.672459 | 0.751004 | 0.845184 | 0.030811 | validation_platt |
| B4-zs | 0.737658 | 0.731293 | 0.810241 | 0.891801 | 0.072773 | validation_temperature |
| B5 | 0.945797 | 0.856484 | 0.877510 | 0.934990 | 0.034492 | sklearn LogisticRegression |
| B6 | 0.838509 | 0.830414 | 0.869478 | 0.927293 | 0.126700 | validation_temperature |

## Score Files
- B0: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B0/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B0/validation.jsonl`
- B0k: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B0k/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B0k/validation.jsonl`
- B1: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B1/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B1/validation.jsonl`
- B2: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B2/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B2/validation.jsonl`
- B2b: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B2b/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B2b/validation.jsonl`
- B3: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B3/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B3/validation.jsonl`
- B4-zs: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B4-zs/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B4-zs/validation.jsonl`
- B5: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B5/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B5/validation.jsonl`
- B6: train `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B6/train.jsonl`; validation `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines/scores/B6/validation.jsonl`

## Validation Stratification
See `validation_stratified_metrics.json` in the output directory.
