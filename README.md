# RL vs. sampling-based motion planning in Drake

Code of my master's thesis at Ostfalia University of Applied Sciences (M.Eng. Automotive Engineering, 2026, grade 2.0).
Thesis title: *Implementierung einer Reinforcement-Learning-basierten Bahnplanung für ein simuliertes 3D-Robotersystem
im Vergleich zu klassischen Planungsverfahren*.

The comparison is between a **SAC step policy** (Stable-Baselines3) and OMPL's **RRT, RRT\* and RRT-Connect**. They run on a
simulated serial arm in a vertical plane, with three joints (3R) or two joints (2R), among walls with openings.

The comparison is fair because every path, learned or sampled, goes through the **same chain**:

```
plan -> shortcut -> KTO -> TOPPRA time scaling -> clearance certificate + limit checks -> Drake simulation (reach and hold)
```

A goal only counts once the simulated arm actually reaches and holds it. A browser dashboard (Meshcat) shows all four
methods side by side on the same request.

## Results (3R, 512 new queries, mean of three runs)

| Method | Approved references |
|---|---|
| SAC policy | **80.2 %** |
| RRT-Connect | 76.7 % |
| RRT | 76.3 % |
| RRT\* | 52.5 % |

- **Policy vs. RRT-Connect:** a paired difference of 3.5 percentage points. All three 95 % intervals lie above zero, but
  the exact McNemar test is not significant in every run.
- **Jointly approved queries:** on these, the policy's references are **11.6 % shorter in duration** and 7.9 % shorter
  in tool path. The final distance to the goal is larger (18.3 vs. 15.0 mm).
- **Scope:** the findings hold for the tested models, scenes and budgets.

## What is in this repository

| Path | Content |
|---|---|
| `src/planar_physics/` | arm model (URDF), scenes, inverse kinematics, collision and clearance certificates, OMPL planners |
| `src/planar_dynamics/` | KTO, TOPPRA trajectory, limits, mission pipeline, Drake rollout |
| `src/planar_learn/` | SAC step policy and its observation contract |
| `src/planar_eval/` | evaluation support (task banks, policy loading, metrics) |
| `src/dashboard/` | web dashboard with Meshcat viewers, training panel, pose maps |
| `scripts/training/` | task bank builder, training environment, SAC training |
| `scripts/evaluation/` | selection, confirmation and final comparison runs, summaries, cost per network width |
| `Dockerfile`, `compose.yaml` | reproducible container: OMPL built from a checksummed source archive, pinned dependencies |

**Not included** (artifacts): trained policies, task banks, training runs and measurement data. The task banks can be
rebuilt with `scripts/training/build_bank.py`, and policies can be trained with `scripts/training/train_kto.py` (see below).

## Quick start

Requires Docker with Compose ≥ 2.39.

```sh
cp .env.example .env              # optional: ports and bind address
docker compose up --build -d      # first build compiles OMPL, later builds reuse the image
```

- 3R dashboard: <http://localhost:4880>
- 2R dashboard: <http://localhost:4881>

The classical planners work right away. For the policy arm you need a trained checkpoint, either from the training
panel or by mounting your own `policy/` folder (see `compose.yaml`).

Container hardening:
- runs as an unprivileged user with a read-only root filesystem;
- no capabilities, `no-new-privileges`, an init process;
- every port is bound to `127.0.0.1` by default.

The endpoints compute without authentication, so expose them only behind your own authenticated HTTPS proxy.

## Building task banks and training

All commands run inside the running dashboard container. Stop dashboard runs first, because they distort timings.
Every script supports `--help`.

```sh
# task bank for the 3R arm (scene, start, goal, IK goals, classical references)
docker compose exec -T dashboard python scripts/training/build_bank.py --output banks/train-3r.json --workers 4

# short training run (a smoke test, not a usable policy)
docker compose exec -T dashboard python scripts/training/train_kto.py \
  --bank banks/train-3r.json --name quick-3r --steps 10000 --checkpoint-every 2000 \
  --envs 5 --torch-threads 2 --width 64 --seed 963 --out runs/quick-3r

# one comparison stage and its summary
docker compose exec -T dashboard python scripts/evaluation/confirm.py \
  --candidate runs/quick-3r/quick-3r --output runs/check-3r --seed 90000000 --count 2
docker compose exec -T dashboard python scripts/evaluation/summarise_final.py runs/check-3r
```

For the 2R arm, add `-e PLANAR2D_ROBOT=2r` and use `banks/train-2r-aperture.json`.

The thesis policies were trained with width 256:
- 3R for 2 million steps;
- 2R for 3 million steps on the scene profile `2r-aperture-clearance-v4`.

The final comparison (`scripts/evaluation/final_runs.py`) runs three times in fresh processes on the same requests:
512 queries for 3R, 256 for 2R. Before it reuses results, it verifies the checksums of checkpoint, manifest and requests.

## Measurement setup

- **IK:** solved once per request, outside the clocks. All methods get the same stored SNOPT goal set.
- **Final evaluation:** planning budget 3 s, mission deadline 20 s, goal tolerance 30 mm, simulation step 2 ms.
- **Planners:** OMPL plans in the bounded joint space with default parameters. The policy first tries straight
  connections to the IK goals, then certified steps with an intermediate goal in the wall opening.
- **Separate results:** reference approval and a held goal are counted separately.
- **No filtering:** all drawn requests are scored, without connectivity or success filters. Training, selection
  (96 requests) and final sets are disjoint and frozen.
- **Model limits:** model parameters are simulation assumptions, not calibrated hardware values.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `DASHBOARD_BIND` | `127.0.0.1` | bind address of all ports |
| `DASHBOARD_PORT` / `DASHBOARD_2R_PORT` | `4880` / `4881` | HTTP 3R / 2R |
| `DASHBOARD_MESH_START` / `_RANGE` | `7800` / `7800-7820` | Meshcat pool 3R |
| `DASHBOARD_2R_MESH_START` / `_RANGE` | `7840` / `7840-7860` | Meshcat pool 2R |
| `BUILD_JOBS` | `3` | compiler parallelism (lower it with little RAM) |
| `PLANAR2D_CHECKPOINT_DIRS` | – | extra checkpoint folders, separated by `:` |

Platforms: Linux AMD64 and ARM64, including Docker Desktop on macOS and Windows (WSL2). About 8 GB RAM are needed for
the build and a short training run.

## Stack

Python 3.14, Drake 1.57, OMPL 2.0.2, Stable-Baselines3 2.9, Gymnasium 1.3, PyTorch 2.11 (CPU), NumPy, Matplotlib, Docker.
