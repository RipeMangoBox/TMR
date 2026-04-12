# Stage4.1 Real-Data 5090 Tmux Launch

Remote repo:

- `/data/public/ripemangobox/Motion/TMR`

Canonical corrected dataset:

- `/data/public/ripemangobox/Motion/datasets/HumanML3D-E`

Compatibility links already point the original absolute paths to the remote roots:

- `/home/ripemangobox/Coding/Github/Motion/TMR`
- `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`

Default corrected run prefix:

- `stage4_1_realdata_e50_b128`

The four tmux sessions are:

1. `s41_d1`
2. `s41_d1_5`
3. `s41_d2a`
4. `s41_d2b`

Launch commands:

```bash
cd /data/public/ripemangobox/Motion/TMR

tmux new-session -d -s s41_d1 \
  'cd /data/public/ripemangobox/Motion/TMR && bash scripts/run_stage4_1_realdata_5090_d1.sh'

tmux new-session -d -s s41_d1_5 \
  'cd /data/public/ripemangobox/Motion/TMR && bash scripts/run_stage4_1_realdata_5090_d1_5.sh'

tmux new-session -d -s s41_d2a \
  'cd /data/public/ripemangobox/Motion/TMR && bash scripts/run_stage4_1_realdata_5090_d2a_wait.sh'

tmux new-session -d -s s41_d2b \
  'cd /data/public/ripemangobox/Motion/TMR && bash scripts/run_stage4_1_realdata_5090_d2b_wait.sh'
```

Useful monitor commands:

```bash
tmux ls
tmux attach -t s41_d1
tmux attach -t s41_d1_5
tmux attach -t s41_d2a
tmux attach -t s41_d2b
```

Logs:

- `logs/stage4_1_realdata_5090/stage4_1_realdata_e50_b128_d1_gpu0.log`
- `logs/stage4_1_realdata_5090/stage4_1_realdata_e50_b128_d1_5_gpu1.log`
- `logs/stage4_1_realdata_5090/stage4_1_realdata_e50_b128_d2a_gpu2.log`
- `logs/stage4_1_realdata_5090/stage4_1_realdata_e50_b128_d2b_gpu3.log`

Expected run dirs:

- `RUN_DIR/stage4_1_realdata_e50_b128_d0`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1_5`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2a`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2b`

After the four remote runs finish, pull the entire corrected run dirs back to
the local machine with:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR
bash scripts/pull_stage4_1_realdata_5090_rundirs.sh
```

Then run local eval + summary:

```bash
bash scripts/eval_stage4_1_realdata_all.sh \
  --run-prefix stage4_1_realdata_e50_b128 \
  --report-date 2026-04-12 \
  --retrieval-batch-size 256
```
