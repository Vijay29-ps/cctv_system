# Incident Decider Architecture

## Goal
Fuse outputs from:
- `snatching_pipeline`
- `fight_weapon_pipeline`

into one final decision:
- `none`
- `snatching`
- `fight_weapon`
- `both`

## Runtime design
1. Pipelines run (mode: `all`, `snatching_only`, or `fight_weapon_only`).
2. `utils.decider_features.build_decider_features(...)` converts pipeline metadata into a fixed feature vector.
3. `utils.main_decider.decide_incidents(...)`:
   - loads trained model from `models/incident_decider_v1.json` (or `DECIDER_MODEL_PATH`)
   - predicts class + confidence
   - if confidence is below threshold (`DECIDER_MIN_CONFIDENCE`), uses conservative heuristic fallback
4. `official/result.json` stores:
   - predicted class and confidence
   - per-class probabilities
   - full decider feature vector
   - reason string (model vs fallback path)

## Model type
Current decider is a **multiclass softmax linear model** trained on fused numeric features.

Why:
- fast and stable on small CCTV datasets
- easy to retrain frequently
- interpretable and easy to deploy (JSON weights, no heavy runtime dependency)

## Feature schema
From both branches:
- `sn_score`
- `sn_locked`
- `sn_confirmed`
- `sn_candidates`
- `sn_rejected`
- `sn_candidate_margin`
- `sn_accept_ratio`
- `fw_score`
- `fight_events`
- `weapon_frames`
- `pose_verified_fight_frames`
- `pose_rejected_fight_boxes`
- `fight_pose_accept_ratio`
- `weapon_to_fight_ratio`

## Training flow
1. Create labeled CSV using this schema + `target` column.
2. Allowed target labels: `none`, `snatching`, `fight_weapon`, `both`.
3. Train:

```bash
python scripts/train_incident_decider.py --train-csv decider_training_template.csv --model-out models/incident_decider_v1.json
```

4. Deploy by placing model at `models/incident_decider_v1.json`.
5. Run system in `PIPELINE_MODE=all`.

## Phase-1 deployment gate (incident label)
- Runtime label is binary: `INCIDENT` / `NO_INCIDENT`.
- Runtime confidence gate uses `DECIDER_PHASE1_INCIDENT_THRESHOLD` (default `0.55`).
- Validate 70%+ accuracy on your real labeled validation set before police deployment:

```bash
python scripts/evaluate_phase1_accuracy.py --csv your_validation.csv --model models/incident_decider_v1.json --label-col target --threshold 0.55 --min-accuracy 0.70
```

## Bootstrap assets included
- Pretrained bootstrap dataset: `data/bootstrap_incident_decider.csv`
- Pretrained model from that dataset: `models/incident_decider_v1.json`
- Training report: `models/incident_decider_v1.metrics.json`

You can use these directly if you do not have your own labeled data yet.
This bootstrap set is synthetic and should be replaced over time with your real CCTV labels.

## Notes
- Keep adding hard negatives (normal walking, crowding, edge boxes, parked vehicles).
- Re-train regularly as class balance changes.
- If dataset grows large, next upgrade path is a temporal model (e.g., TCN/LSTM) on windowed frame-level features.
