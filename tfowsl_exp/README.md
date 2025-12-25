# TF-OWSL Experiment Scaffold (Config + Delay/Fer modes)

## Run (toy)
python3 -m exp.run --config configs/toy.yaml

## Run (jsonl stream)
python3 -m exp.run --config configs/default.yaml

Outputs:
- outputs/<run>/logs/*.jsonl
- outputs/<run>/events_<method>.json
- outputs/<run>/summary.csv
- outputs/<run>/plots/*.png

Config:
- metrics.delay.mode: domain|intent (default domain)
- metrics.fer.mode: oos_trigger|mixture (default oos_trigger)
