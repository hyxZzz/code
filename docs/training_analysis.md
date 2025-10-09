# Training diagnostics for modified attacker guidance

## Run summary (`runs/default`)

- `eval/success` peaked at **5 %** around update 340 and finished at **3 %** at
  update 400, indicating that the defenders almost never complete the escort
  task successfully under the current dynamics.
- `train/ep_return` decreased from a high of roughly **15.3** near update 161 to
  **9.5** by the end of training, while `train/success` fell back to zero – the
  policy no longer discovers successful trajectories late in training.
- The imitation loss remained high (>1.3) even at the end of training,
  signalling that the teacher demonstrations are struggling with the new,
  faster attackers and therefore provide weak residual guidance.

These observations explain the poor final performance after switching the
attackers (`P`) to the predictive lead-pursuit guidance law.

## Remediation implemented

- Upgraded the handcrafted teacher controller to blend a proportional
  navigation (PN) term with the existing PD range-keeping action. The PN term is
  only applied when the attacker is closing in, producing lateral acceleration
  commands that better match the manoeuvring requirements imposed by the
  lead-pursuit attackers.
- Documented the attacker guidance behaviour directly in the environment code
  for clarity.

The stronger teacher signal should reduce imitation loss and give PPO higher
-quality residual targets, while still respecting the acceleration budget via
`clamp_norm`.
