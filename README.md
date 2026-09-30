# Adaptive Variable-Resolution 2.5D LiDAR Mapping

**SIH26053 — Project-Phase Implementation**

> **Resolution is allocated by risk, not distance.**

A library-first research and engineering implementation of a LiDAR mapping pipeline that dynamically allocates spatial resolution according to **range, semantic relevance, dynamic behavior, object-relative risk, and predicted motion**, while respecting a per-frame compute budget.

This repository contains the algorithmic core only:

- LiDAR data sources
- semantic perception
- adaptive 2.5D quadtree mapping
- risk-aware resolution control
- dynamic detection and tracking
- prediction and risk bubbles
- deterministic compute-budget enforcement
- benchmarks and tests

**There is no frontend, HTTP server, or visualization layer in this repository.**

---

## Table of Contents

- [Core Idea](#core-idea)
- [What the Repository Implements](#what-the-repository-implements)
- [Architecture](#architecture)
- [Resolution Policies](#resolution-policies)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Examples and Benchmarks](#examples-and-benchmarks)
- [Measured Benchmark](#measured-benchmark)
- [Compute Budget](#compute-budget)
- [Risk Bubble](#risk-bubble)
- [Semantic Provenance](#semantic-provenance)
- [RandLA-Net](#randla-net)
- [SemanticKITTI](#semantickitti)
- [Tests](#tests)
- [Repository Layout](#repository-layout)
- [Design Details](#design-details)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Core Idea

Traditional variable-resolution LiDAR mapping can allocate resolution primarily according to distance:

```text
Resolution = f(distance)
```

This project treats resolution as a resource that should instead be allocated according to the importance of each region:

```text
                         ┌──────────────┐
                         │   LiDAR      │
                         └──────┬───────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
          Range             Semantics          Dynamics
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                           Prediction
                                │
                                ▼
                       Importance / Risk
                                │
                                ▼
                       Resolution Control
                                │
                                ▼
                         Adaptive Map
```

The adaptive policy combines five importance factors:

1. **Range**
2. **Semantic relevance**
3. **Dynamic relevance**
4. **Object-relative bubble relevance**
5. **Predicted future relevance**

The result is an adaptive quadtree in which different regions can have genuinely different spatial resolutions.

> **The goal is not to make the entire map maximally detailed. The goal is to make the right parts of the map detailed at the right time.**

---

# What the Repository Implements

The current implementation provides the following pipeline:

1. **Preprocessing**
   - Range filtering
   - Ego-frame convention (`+x` forward)

2. **Semantic labelling**
   - RandLA-Net when a usable checkpoint and PyTorch are available
   - Dataset ground-truth labels when configured
   - `unknown` fallback when neither is available

3. **Dynamic analysis**
   - Temporal occupancy/height change detection
   - Decaying dynamic heat

4. **Object tracking**
   - Change-driven object detection
   - Association
   - Constant-velocity Kalman filters

5. **Prediction**
   - Risk bubbles
   - Constant-velocity prediction corridors

6. **Resolution control**
   - Five importance factors
   - Hysteresis
   - Target cell size

7. **2.5D mapping**
   - Variable-resolution quadtree leaves
   - Per-cell spatial statistics

8. **Budget enforcement**
   - Deterministic lowest-importance-first coarsening
   - Coarsening only; cells are not deleted as a budget operation

Three policies can process the **same input frame**:

```text
uniform
   │
   └── one cell size everywhere

distance
   │
   └── resolution controlled by range

adaptive
   │
   └── range
       + semantic relevance
       + dynamic relevance
       + risk bubbles
       + prediction
```

---

# Architecture

```text
                         Point Cloud
                              │
                              ▼
                    ┌───────────────────┐
                    │   Preprocessing   │
                    │ range / ego frame │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Semantic Perception│
                    │ RandLA-Net / GT / │
                    │      unknown      │
                    └─────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
       Temporal Analysis                Semantic State
               │
               ▼
       Dynamic Detection
               │
               ▼
       Object Association
               │
               ▼
       Kalman CV Tracking
               │
               ▼
        Velocity / Prediction
               │
               └──────────────┐
                              ▼
                    ┌───────────────────┐
                    │ Importance Model  │
                    │                   │
                    │ Range             │
                    │ Semantic          │
                    │ Dynamic           │
                    │ Bubble            │
                    │ Prediction        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Resolution Control│
                    │ split / keep /    │
                    │ merge + hysteresis│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ 2.5D Quadtree     │
                    │ variable leaves   │
                    │ + cell statistics │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Compute Budget    │
                    │ lowest-importance │
                    │ first coarsening  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   FrameResult     │
                    │ map + tracks +    │
                    │ metrics + debug   │
                    └───────────────────┘
```

---

# Resolution Policies

## 1. Uniform

Every region uses the same spatial resolution.

```text
┌─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┤
└─┴─┴─┴─┴─┴─┴─┘
```

This provides the reference case for resource usage.

---

## 2. Distance

Resolution depends on range from the ego vehicle.

Conceptually:

```text
Near ego                         Far from ego

██████████████                  ███    ███
██████████████                  ███    ███
██████████████                  ███    ███
```

This reduces representation away from the ego vehicle but does not directly account for semantics, motion, or predicted relevance.

---

## 3. Adaptive

The adaptive policy combines five factors:

```text
                  ┌────────────┐
                  │   Range    │
                  └─────┬──────┘
                        │
                  ┌─────▼──────┐
                  │  Semantic  │
                  └─────┬──────┘
                        │
                  ┌─────▼──────┐
                  │  Dynamic   │
                  └─────┬──────┘
                        │
                  ┌─────▼──────┐
                  │   Bubble   │
                  └─────┬──────┘
                        │
                  ┌─────▼──────┐
                  │ Prediction │
                  └─────┬──────┘
                        │
                        ▼
                  Importance Score
                        │
                        ▼
                 Quadtree Resolution
```

This means a distant dynamic object can receive finer representation than a nearby low-priority static region.

---

# Installation

Requires **Python >= 3.10**.

The core package requires:

- `numpy`
- `pyyaml`

Install the repository in editable mode:

```bash
pip install -e .
```

### Optional RandLA-Net support

PyTorch is optional.

Install the optional RandLA-Net dependency group with:

```bash
pip install .[randlanet]
```

Without PyTorch and a usable checkpoint, the pipeline does **not** pretend to run neural inference. It falls back according to the configured semantic source.

See:

- `docs/RANDLANET.md`
- `models/randlanet/README.md`

for the RandLA-Net integration details.

---

# Quick Start

```python
from adaptive_lidar import AdaptiveLidarMapper
from adaptive_lidar.data import SyntheticPointCloudSource

source = SyntheticPointCloudSource(
    scenario="urban_crossing",
    num_frames=20,
)

mapper = AdaptiveLidarMapper("configs/default.yaml")

for i in range(len(source)):
    result = mapper.process_frame(source.load_frame(i))

    print(
        result.metrics.total_cells,
        result.metrics.frame_latency,
    )
```

`AdaptiveLidarMapper` accepts:

- a `PointCloud`
- an `(N, 3)` NumPy array
- an `(N, 4)` NumPy array
- a list of `LiDARPoint`

and returns a `FrameResult` containing the resulting quadtree leaves, tracks, measured metrics, and diagnostics.

A `PipelineConfig` can also be supplied instead of a YAML path.

---

# Command Line Interface

The package exposes both module and console-script interfaces.

### Run the pipeline

```bash
python -m adaptive_lidar run --frames 20
```

### Compare policies

```bash
python -m adaptive_lidar compare --frames 20
```

Runs the three policies on the same input frames.

### Run importance ablations

```bash
python -m adaptive_lidar ablate --frames 20
```

### Run budget sweep

```bash
python -m adaptive_lidar budget --frames 20
```

### Inspect available data sources

```bash
python -m adaptive_lidar info
```

### Inspect resolved configuration and capabilities

```bash
python -m adaptive_lidar status
```

The installed console script is also available:

```bash
adaptive-lidar --help
```

---

# Examples and Benchmarks

Runnable examples:

```bash
python -m examples.basic_mapping
```

Per-frame measured table.

```bash
python -m examples.compare_modes
```

Uniform / distance / adaptive comparison.

```bash
python -m examples.ablation
```

Importance-factor ablation.

```bash
python -m examples.budget
```

Compute-budget sweep.

---

## Benchmark commands

### Frame benchmark

```bash
python -m benchmarks.bench_frame
```

### Policy comparison

```bash
python -m benchmarks.bench_modes
```

### Budget sweep

```bash
python -m benchmarks.bench_budget
```

Each benchmark writes:

```text
results/<name>.json
results/<name>.csv
```

Regenerate benchmark artifacts before citing them:

```bash
python -m benchmarks.bench_frame --frames 30
python -m benchmarks.bench_modes --frames 20
python -m benchmarks.bench_budget --frames 8
```

> **All timing and usage numbers are wall-clock measurements from the machine that ran the benchmark.**

---

# Measured Benchmark

The repository currently contains a measured synthetic benchmark from the `urban_crossing` scenario.

**Frame 19 of 20:**

| Mode | Cells | High | Medium | Coarse | Estimated Memory | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 4,972 | 4,972 | 0 | 0 | 1.05 MB | 113 ms |
| Distance | 3,773 | 47 | 492 | 3,234 | 0.57 MB | 103 ms |
| Adaptive | 5,095 | 2,527 | 2,550 | 18 | 1.04 MB | 140 ms |

These values are **representative measured results from the repository's synthetic benchmark**, not universal performance guarantees.

The result illustrates an important distinction:

- **Uniform** maintains fine resolution everywhere.
- **Distance** aggressively coarsens regions according to range.
- **Adaptive** spends substantially more of its representation on high-importance regions.

Therefore, the adaptive policy should **not** be interpreted as simply minimizing the number of cells.

Its objective is risk-aware allocation of spatial resolution.

---

# Compute Budget

The budget stage enforces a per-frame target by **coarsening low-importance regions first**.

The important invariant is:

> **Budget enforcement coarsens the representation; it does not delete map content as a budget operation.**

Conceptually:

```text
              Budget available
                     │
                     ▼
           ┌───────────────────┐
           │ Current quadtree  │
           └─────────┬─────────┘
                     │
             Cost > target?
                /         \
              no           yes
              │             │
              ▼             ▼
            Keep      Find lowest
                      importance
                           │
                           ▼
                        Coarsen
                           │
                           ▼
                    Re-check budget
```

The repository includes four budget presets:

```text
100%
75%
50%
25%
```

The budget stage keeps all four presets feasible in the current benchmark and only coarsens.

See:

```text
results/bench_budget.json
```

for the measured post-budget cost and merge counts.

---

# Risk Bubble

A tracked object is surrounded by an object-relative risk bubble.

The implemented relationship is:

\[
r_{\text{bubble}}
=
r_{\min}
+
\kappa
\|\mathbf{v}\|
T
\]

where:

- \(r_{\text{bubble}}\) is the bubble radius
- \(r_{\min}\) is the class-dependent minimum radius
- \(\|\mathbf{v}\|\) is object speed
- \(T\) is the prediction horizon
- \(\kappa\) controls velocity scaling

Current class minima:

| Object class | Minimum radius |
|---|---:|
| Pedestrian | 2 m |
| Cyclist | 3 m |
| Vehicle | 5 m |

Current defaults:

```text
kappa = 1.5
T     = 2 s
```

Cells inside an active bubble are protected from budget coarsening together with the required ancestor chain up to the configured floor level.

This allows high-risk regions around tracked dynamic objects to retain resolution when the rest of the map is being coarsened.

---

# Semantic Provenance

The repository explicitly distinguishes the source of semantic labels.

Every frame records one of:

```text
ground_truth
model_prediction
unknown
```

Ground-truth labels are **never reported as model predictions**.

This distinction is important when evaluating or presenting the pipeline.

The intended provenance flow is:

```text
SemanticKITTI GT
      │
      └── semantic_source = ground_truth


RandLA-Net inference
      │
      └── semantic_source = model_prediction


No usable semantic source
      │
      └── semantic_source = unknown
```

---

# RandLA-Net

RandLA-Net is the optional semantic-perception component.

Its responsibility is to provide point-wise semantic information to the mapping and resolution-control stages.

The conceptual flow is:

```text
LiDAR point cloud
       │
       ▼
    RandLA-Net
       │
       ▼
Point-wise semantic labels
       │
       ▼
2.5D map cells
       │
       ▼
Semantic importance
       │
       ▼
Resolution controller
```

RandLA-Net is **not itself the resolution-allocation algorithm**.

The mapping/controller system consumes semantic information as one of the inputs to the adaptive policy.

### Checkpoint status

No RandLA-Net checkpoint is shipped with this repository.

If `use_model=True` is requested without PyTorch or without a matching checkpoint, the implementation raises:

```text
CheckpointNotAvailable
```

The graph builder is unit-tested without PyTorch. The model forward/shape check is gated on PyTorch availability.

See:

```text
docs/RANDLANET.md
models/randlanet/README.md
```

for the current integration details.

---

# SemanticKITTI

SemanticKITTI is supported as a real LiDAR data source.

The source is enabled when:

```text
KITTI_ROOT
```

points to a valid SemanticKITTI installation.

The intended real-data path is:

```text
SemanticKITTI .bin frame
          │
          ▼
     Point cloud
          │
          ▼
     Preprocessing
          │
          ▼
 Semantic labels / GT
          │
          ▼
    2.5D projection
          │
          ▼
 Adaptive quadtree
          │
          ▼
 Risk-aware controller
```

For CI and lightweight development, prefixed sequence/per-frame samples are used instead of requiring the full dataset.

A complete SemanticKITTI installation is therefore required for full real-dataset execution.

---

# Tests

Run the complete test suite with:

```bash
python -m pytest
```

The repository currently has **150+ tests passing**.

Two cases are skipped when PyTorch is unavailable because they require the neural-network forward/shape path.

The test suite covers the core mapping, control, data, tracking, provenance, and optional perception interfaces.

---

# Repository Layout

```text
.
├── src/
│   └── adaptive_lidar/
│       ├── data/
│       │   ├── synthetic.py
│       │   └── SemanticKITTI sources
│       │
│       ├── perception/
│       │   ├── semantic labelling
│       │   ├── dynamics
│       │   └── RandLA-Net integration
│       │
│       ├── mapping/
│       │   ├── map_cell.py
│       │   ├── 2.5D projection
│       │   └── quadtree
│       │
│       ├── controller/
│       │   ├── importance
│       │   ├── hysteresis
│       │   └── budget manager
│       │
│       ├── tracking/
│       │   ├── object tracker
│       │   └── Kalman prediction
│       │
│       ├── prediction/
│       │   ├── trajectory models
│       │   └── risk bubbles
│       │
│       ├── pipeline.py
│       ├── config.py
│       └── __main__.py
│
├── configs/
│   └── default.yaml
│
├── examples/
│
├── benchmarks/
│
├── results/
│
├── tests/
│
├── docs/
│   └── RANDLANET.md
│
├── models/
│   └── randlanet/
│       └── README.md
│
├── pyproject.toml
└── README.md
```

---

# Design Details

## 2.5D representation

Each map cell maintains spatial statistics rather than a full 3D voxel volume.

The representation can capture quantities such as:

- height statistics
- occupancy
- semantic state
- dynamic state
- importance
- resolution level

This provides a compact ground-plane representation while retaining vertical information relevant to navigation.

---

## Adaptive quadtree

The map is represented by hierarchical spatial cells.

A cell can be:

```text
coarse
  ↓
split
  ↓
medium
  ↓
split
  ↓
fine
```

The important property is that adaptive resolution changes the **actual spatial leaf structure**.

It is not merely a fixed grid with different colors.

---

## Importance-driven control

The adaptive controller combines:

```text
Range
  +
Semantic
  +
Dynamic
  +
Bubble
  +
Prediction
  ↓
Importance
  ↓
Target resolution
  ↓
Quadtree split / keep / merge
```

The controller also applies hysteresis so that cells near decision thresholds do not continuously oscillate between refinement and coarsening.

---

## Tracking and prediction

The dynamic pipeline is:

```text
Temporal change
      ↓
Dynamic cells
      ↓
Object detection / association
      ↓
KalmanCV tracking
      ↓
Velocity estimate
      ↓
Prediction corridor
      ↓
Risk bubble
      ↓
Resolution protection
```

This gives the controller temporal context rather than making an independent resolution decision from a single frame.

---

# Limitations

The current implementation has several known limitations.

### 1. Low-budget performance

Budget enforcement at:

```text
<= 50%
```

is currently dominated by a Python merge loop rather than map semantics.

This is documented as future work.

---

### 2. RandLA-Net execution

RandLA-Net is implemented as an optional integration, but the neural inference path has not been fully exercised in this repository because PyTorch is not installed in the current test environment.

The graph builder is unit-tested without PyTorch, while the model forward/shape check is gated on PyTorch availability.

---

### 3. SemanticKITTI dependency

Full SemanticKITTI execution requires:

```text
KITTI_ROOT
```

to point to a real SemanticKITTI dataset.

The repository uses prefixed sequence/per-frame samples for CI so that tests do not require the complete dataset.

---

### 4. Benchmark hardware dependence

Latency and resource measurements depend on the machine used to run the benchmark.

They should therefore be reported with:

- hardware
- Python version
- dependency versions
- number of frames
- scenario
- configuration

rather than treated as universal performance values.

---

### 5. Memory estimate

The repository's memory values are **structural estimates**, not process RSS measurements.

The estimate is based on the documented `MapCell` footprint:

```text
CELL_BYTES = 64
```

plus implied internal nodes.

These numbers are labelled **Estimated** for this reason.

---

### 6. No accuracy claim

This repository does **not** claim a semantic segmentation accuracy number.

Semantic ground truth and model predictions are explicitly separated through provenance metadata.

---

# Reproducibility

Before citing benchmark numbers, regenerate them:

```bash
python -m benchmarks.bench_frame --frames 30
python -m benchmarks.bench_modes --frames 20
python -m benchmarks.bench_budget --frames 8
```

For a reproducible experiment, record:

```text
Dataset / scenario
Frame range
Policy
Configuration
Importance weights
Budget
Quadtree limits
Semantic source
RandLA-Net checkpoint, if used
Python version
Dependency versions
Hardware
```

The repository's benchmark artifacts are snapshots of the machine on which they were generated.

---

# Citation

If you use this repository in research, presentations, or technical reports, cite the project as:

```bibtex
@software{adaptive_variable_resolution_lidar,
  title  = {Adaptive Variable-Resolution 2.5D LiDAR Mapping},
  author = {Prayash Bhagawati},
  year   = {2026},
  note   = {SIH26053 project-phase research and engineering implementation}
}
```

---

# License

Add the project's actual license before publishing the repository.

For example, if the project is intended to use the MIT License:

```text
MIT License
```

A license should only be claimed when the corresponding `LICENSE` file is included in the repository.

---

# The One-Sentence Version

> **Every frame, the system evaluates how important each LiDAR map region is and allocates its spatial-resolution budget accordingly—using range, semantics, dynamics, object-relative risk bubbles, and predicted motion rather than distance alone.**

---

## Core Principle

```text
Traditional distance-based mapping:

        Resolution = f(distance)


This project:

        Resolution = f(
            range,
            semantic relevance,
            dynamic behavior,
            risk bubble,
            predicted relevance,
            compute budget
        )
```

**We do not map everything equally. We allocate resolution where it matters.**
