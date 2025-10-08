# 3D Pursuit Hierarchical RL Project

This repository contains the hierarchical reinforcement learning stack used in the
3D pursuit/escort environment.  The workflow is split into two stages:

1. Train the low-level residual controller that outputs acceleration residuals
   for each defender while the manager follows the built-in rule-based
   assignment.
2. Optimise the high-level manager on top of the frozen controller to learn
   defender-to-attacker assignments.

The sections below list the commands required to run the full project end to
end.

## 1. Environment setup

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install runtime dependencies
pip install torch numpy pyyaml matplotlib imageio
```

These packages are sufficient for training, evaluation, log analysis, and GIF
visualisation.  Installing `scipy` is recommended when using the Hungarian
matcher implementation.

## 2. Train the residual controller

```bash
python -m src.train --config src/configs/default.yaml
```

This command launches PPO training for the defender residual controller while
the manager runs in `rule` mode (see `manager.mode` in the config).  The script
writes TensorBoard-style scalars to `runs/default` and stores checkpoints in
`ckpts/best.pt` (best validation success rate) and `ckpts/latest.pt` (last
update).  The `best.pt` checkpoint is later reused when training the manager.

## 3. Evaluate controller checkpoints

```bash
python -m src.validate --config src/configs/default.yaml --episodes 1000
```

Running the validator reproduces the success rate over a large number of
episodes for both `best.pt` and `latest.pt`, confirming the quality of the base
policy before introducing the manager layer.

## 4. Train the high-level manager

```bash
python -m src.train_manager --config src/configs/default.yaml
```

Before launching the command, ensure that `manager_train.controller_ckpt`
points to the trained controller checkpoint (default: `ckpts/best.pt`).  The
manager script automatically switches the environment to `manager.mode =
"learned"`, loads the frozen controller to produce residual actions, and
optimises the categorical actor-critic with PPO.  Logs are saved to
`runs/default/manager`, and the best weights are written to
`ckpts/manager_best.pt`.

### Why sequential training?

- **Stability:** learning the low-level residual controller first avoids the
  non-stationarity that would arise if the manager kept changing assignments
  while the controller was still exploring.
- **Sample efficiency:** the manager trains against a strong, fixed controller,
  so credit assignment focuses on high-level decisions instead of compensating
  for an under-trained low-level policy.
- **Modularity:** the trained controller can still be evaluated or deployed
  with the rule-based manager when needed, while the learned manager can be
  plugged in later to boost coordination.

## 5. Evaluate the full hierarchical stack

```bash
python -m src.eval_hierarchical \
  --config src/configs/default.yaml \
  --controller-ckpt ckpts/best.pt \
  --manager-ckpt ckpts/manager_best.pt \
  --episodes 300
```

This evaluation script loads both checkpoints, rolls out multiple episodes with
the manager issuing assignments and the controller producing residual actions,
and prints the final success rate.

## 6. Inspect training logs

```bash
python analyze_scalars.py runs/default/scalars.json
```

The helper script summarises metric trends (returns, success rate, imitation
loss) and reports whether checkpoints are present.

## 7. Visualise trajectories

```bash
python -m src.viz_gif --config src/configs/default.yaml \
  --weights ckpts/best.pt --out runs/viz_best.gif --frames 600 --fps 15
```

The renderer generates a GIF containing the target, defenders, attackers, and
assignment lines, allowing qualitative inspection of the learned behaviour.

