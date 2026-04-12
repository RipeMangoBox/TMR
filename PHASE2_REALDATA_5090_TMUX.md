# Phase2 Real-Data 5090 Tmux Launch

Remote repo:

- `/data/public/ripemangobox/Motion/TMR`

Canonical corrected D2b warm-start:

- `RUN_DIR/stage4_1_realdata_e50_b128_d2b/last_weights`

Default Phase2 family:

- `phase2_realdata_e50_b128`

Current Phase2 suffix:

- `p2a`

Launch command:

```bash
cd /data/public/ripemangobox/Motion/TMR

tmux new-session -d -s p2_p2a \
  'cd /data/public/ripemangobox/Motion/TMR && bash scripts/run_phase2_realdata_5090_p2a.sh'
```

If you want a fresh explicit server suffix instead of reusing `p2a`:

```bash
cd /data/public/ripemangobox/Motion/TMR

tmux new-session -d -s p2_p2a_srv1 \
  "cd /data/public/ripemangobox/Motion/TMR && RUN_PREFIX=phase2_realdata_e50_b128_srv1 PHASE2_SUFFIX=p2a bash scripts/run_phase2_realdata_5090_p2a.sh"
```

Useful monitor commands:

```bash
tmux ls
tmux attach -t p2_p2a
```

Default log file:

- `logs/phase2_realdata_5090/phase2_realdata_e50_b128_p2a_gpu0.log`

After training finishes, run retrieval + report:

```bash
cd /data/public/ripemangobox/Motion/TMR

RUN_DIR=RUN_DIR/phase2_realdata_e50_b128_p2a \
BASE_DIR=RUN_DIR/stage4_1_realdata_e50_b128_d2b \
REPORT_DATE=2026-04-12 \
bash scripts/eval_phase2_realdata.sh
```
