# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Quick start commands
- **Create the project environment (miniconda)**
  ```bash
  make conda-env
  ```
  Creates conda env `fractal-frontier` (Python 3.12 – required because
  numba does not support Python ≥3.13) and installs all dependencies.
  All other targets auto-detect this env; override with
  `CONDA_ENV=<name>` (e.g. `make test CONDA_ENV=bigcodebench`).
- **Install dependencies**
  ```bash
  make install
  ```
  Installs into the detected interpreter (conda env, venv, or system
  python3 when conda is absent).
- **Run the application**
  ```bash
  make run
  ```
  This starts a Tkinter window that displays the Mandelbrot set and allows zooming, panning, theme changes, and bookmark management.
- **Run tests** – The project now includes an automated test suite. Run all tests with:
  ```bash
  make test
  ```
  or run a specific test file:
  ```bash
  pytest tests/test_calculator.py
  ```
- **Linting** – Run flake8 with the project's `.flake8` configuration:
  ```bash
  make lint
  ```
  Settings: max line length 88 (matches black), `tests/` excluded from
  linting, `E203` ignored (conflicts with black's slice spacing).
- **Formatting** – The project uses black and isort for code formatting. Run with:
  ```bash
  make format
  ```
- **Full check** – Format and lint in one pass:
  ```bash
  make check
  ```
- **Environment check** – Show the interpreter and runtime deps the targets will use:
  ```bash
  make env-check
  ```
- **Cleaning** – Remove build artifacts with:
  ```bash
  make clean
  ```
- **Help** – See all available targets with:
  ```bash
  make help
  ```

## Project structure overview
```
Fractal Frontier/            # root of the repository
├── Fractal_Frontier.py      # Main application (GUI, viewer, interaction, tile cache)
├── fractals.py              # CPU/GPU calculator and backend selection
├── gpu_fractals.py          # Numba CUDA kernels and launch configuration
├── themes.py                # Colour themes and palette helpers
├── dd_math.py               # Double-double (quad-float) arithmetic helpers
├── hp_view.py               # Arbitrary-precision view state (PreciseBounds)
├── deep_zoom.py             # Perturbation deep-zoom backend (mpmath references)
├── __init__.py              # Public package API (calculator, JIT kernels, themes)
├── requirements.txt         # Runtime and development dependencies
├── pytest.ini               # Test configuration and warning policy
├── Makefile                 # Environment, test, lint, format, and run targets
├── README.md                # High‑level documentation
├── AGENTS.md                # Architecture and development guidance
├── LICENSE                  # MIT license
├── .flake8                  # Lint configuration
├── .gitignore               # Ignored files (pycache, caches, aider history)
└── tests/                   # Test suite directory
    ├── test_calculator.py
    ├── test_colors.py
    ├── test_gpu.py
    ├── test_deep_zoom.py
    ├── test_hp_view.py       # PreciseBounds unit tests
    ├── test_hp_integration.py # HP view state wired into the viewer
    ├── test_regressions.py
    ├── test_viewer_functionality.py
    └── test_viewer_integration.py
```
The core logic lives in `Fractal_Frontier.py`:
- **Color theme system** – `apply_color_theme()` dispatches to a set of predefined themes (Default, Grayscale, Blue, Fire, Rainbow…). The high-resolution themes (Rainbow2/3/4) use finer HSV palettes with 256/1024/8192 hue steps.
- **JIT‑compiled iteration** – `mandelbrot_section_jit` uses Numba (`@njit(parallel=True)`) to compute the Mandelbrot iterations for a vertical slice.
- **Calculator wrapper** – `MandelbrotCalculator` provides methods for calculating different fractal types:
  - `calculate_mandelbrot_section()` for Mandelbrot set calculations
  - `calculate_julia_section()` for Julia set calculations
  - `calculate_fatou_section()` for Fatou set calculations
- **GUI layer** – `MandelbrotViewer` creates a Tkinter canvas, handles user input (mouse wheel, buttons, keyboard), manages zoom/pan logic, and coordinates multiprocessing via a `multiprocessing.Pool` of worker processes equal to the number of CPU cores. The application now supports multiple fractal types (Mandelbrot, Julia, Fatou) and a CPU Cores theme that randomly selects a color theme for each section.
- **Async rendering** – `draw_mandelbrot()` only enqueues a request; a daemon worker thread computes the image (GPU or process pool) and hands results back through `_result_queue`, which the main thread polls via `_poll_results()`. The worker thread never calls Tk directly. Only the newest request is rendered.
- **Title metrics** – the window title reports the settled render time with three decimal places and a smoothed preview FPS measured from displayed preview-frame intervals; the status bar keeps backend/coordinate details.
- **Pan tile cache** – settled full renders return raw iteration counts. The viewer seeds full-resolution tiles from that array and the worker asynchronously precomputes one ring of adjacent tiles (`_tile_queue`). Drags composite cached tiles and colour-map them on the main thread; missing tiles are black until their worker result arrives. Recent tiles outside the active ring are kept in a bounded LRU history (384 tiles) so a quick left/right reversal reuses them. Tiles themselves are never downscaled: preview display starts at full resolution and only steps down to half/quarter after measured frames exceed the frame budget, recovering after several cheap frames.
- **GPU backend** – `fractals.py` sets `GPU_AVAILABLE` only when a real CUDA device is present (importing numba-cuda alone is not enough). `MandelbrotCalculator._compute_iterations()` falls back to the CPU JIT path if a GPU call fails at runtime, and the viewer syncs its GPU checkbox from the calculator state.
- **Arbitrary-precision view state (`hp_view.py`)** – the canonical view is a `PreciseBounds`: four `Decimal` endpoints plus a `precision` (digit count). Everything else is derived. Two rules keep deep state exact:
  - **All Decimal math runs inside an explicit `localcontext`.** The default context is 28 digits and silently truncates at depth. Mutations use `_mut_ctx()` (exactly the policy precision); reads use the wider `_ctx(*operands, extra)`. A bare `Decimal` operation anywhere in the view path is a bug.
  - **Mutations round *to* the policy precision, never grow it.** `adopt_precision()` runs first and takes `_op_precision()` (derived from the magnitude of the operands), so precision tracks the *window scale*, not the accumulated operations. Without this a zoom chain creeps from 50 digits to hundreds. `HP_PRECISION_CAP` is `None` by default: depth is unbounded, and a guard is available but not imposed.
  `required_precision(pitch)` sizes precision from pitch *magnitude* (`exact_` style), not raw exponent. The DD pairs (`x_min`/`x_min_lo` …) survive as **properties** on the viewer that read from and write through `PreciseBounds`, so legacy kernels, tests, and old code that assigns `viewer.x_max_lo` keep working, and the mirror cannot drift.
- **Deep-zoom backends** – float64 pixel coordinates stop resolving detail once the pixel pitch approaches the coordinate ULP (~1e13 magnification; the symptom is flat color plateaus and block seams). Four depth bands, chosen automatically by `MandelbrotViewer._select_precision()` (compare pitch to ULP with a 64-pixel margin, `DEEP_MARGIN`):
  - `float32`/`float64`: existing GPU/CPU kernels (up to ~1e13).
  - `double_double`: coordinates carried as (hi, lo) double-double sums (`dd_math.py` helpers, Numba mirrors in `fractals.py`), smooth to ~1e30; CPU-only.
  - `perturb`: the Deep Zoom Algorithm (`deep_zoom.py`) – a complex mpmath reference orbit per section plus float64 delta iteration with re-rooting; CPU-only; Mandelbrot only (deep Julia clamps to double-double, Fatou has no deep kernel).
  - `perturb_hp`: the same algorithm fed from `PreciseBounds` (`calculate_mandelbrot_perturb_hp`). Section starts and per-pixel deltas come from the exact Decimal grid, so pixel **identity** no longer depends on DD resolution. `fractals.is_deep()` is the single place that decides what counts as a deep backend.
  The viewer keeps its view bounds as double-double pairs, mutated only through `PreciseBounds`. Tile anchors, pan-cache bounds, and bookmarks carry the DD pairs; bookmark JSON stores `x_min_lo` fields (old files default to 0) plus `x_min_hp`/`view_precision` when saved at depth. Tile caching and pan-cache cropping are **disabled at `perturb_hp` depth** – their anchors are DD, which is exactly what stops resolving there. `mpmath` is a runtime dependency.
  **The ceiling was DD coordinates, not the algorithm.** Measured at bookmark 9 (pitch ~1.6e-34, 1200 px): the stored `lo` has a granularity of 2^-109 (1.5e-33), so the pitch is 0.1x one lo-ULP and the DD path resolves **127 distinct columns out of 600** regardless of width; `PreciseBounds` resolves all 600. Consequence: bookmark 9 cannot be recovered at any render width – the information was never stored. Deep views created *after* this change carry exact coordinates, and the phase-3 acceptance test therefore uses a freshly built HP view rather than bm9, comparing against direct mpmath iteration pixel-for-pixel.
  **Unresolved pixels are measured, not assumed away.** `calculate_mandelbrot_perturb_hp` reports `stats["unresolved"]` and `stats["direct"]`; the fallback runs direct mpmath iteration under a budget (`HP_DIRECT_BUDGET`) and only then treats the remainder as interior. Watch those counters before considering a triple-double kernel.
  **Perturbation glitch detection uses Pauldelbrot's criterion.** `_perturb_counts_jit` checks `|Z_n + δ_n|² < SWITCH_RATIO2 · |Z_n|²` (`SWITCH_RATIO2 = 1e-6`, i.e. float64 cancellation between `Z_n` and `δ_n`) and flags pixels that outlive an early-escaping reference orbit (`n == zs_len < max_iterations`). Never test `|δ_n|² > ratio · |Z_n|²`: because `δ_n` is iterated with the full quadratic recurrence, `|δ_n|` legitimately grows to `O(1)` before a pixel escapes, so an upper bound on `|δ_n|/|Z_n|` flags every non-root pixel as a glitch and forces a slow fallback or black interior output at HP depth.
  **CUDA + cyclic GC in the test suite.** Every test that instantiates `MandelbrotViewer` must call `viewer._close_app()` before `root.destroy()`. Leaving a viewer unclosed leaks its daemon `_render_thread` and 32-worker `multiprocessing.Pool`, and leaves `MandelbrotViewer` / `tk.Variable` / `PhotoImage` / CUDA buffer cycles to be collected asynchronously by Python's cyclic GC on whichever thread triggers a GC pass (including another viewer's `_render_worker` thread inside `launch_kernel` → `safe_cuda_api_call`, which aborts the interpreter). `gpu_fractals._locked` also suspends cyclic GC while `_GPU_LOCK` is held so a GC pass can never fire mid-CUDA-call.
  **`gmpy2` is not a proven poison — but it is not in the environment.** mpmath picks up GMP/MPFR automatically when gmpy2 is importable, which speeds the reference-orbit builder from 10.8 ms to 8.2 ms per 250-digit, 4000-iteration orbit; per-pixel work is float64 Numba either way. It was excluded while the abort above was misattributed to it. The exclusion is now a precaution, not a diagnosis. `deep_zoom.precision_backend()` / `precision_accelerated()` report which backend is live, and `FRACTAL_PRECISION_ACCELERATOR=python` opts out without uninstalling. Note that "the deep bands never launch GPU kernels" does **not** make a C extension safe from CUDA: importing `numba.cuda` creates a CUDA context at import time whenever a GPU is present.
  **Pool workers:** the pool passes an initializer that sets `numba.set_num_threads(1)`. Work is already split across processes, so a per-worker `parallel=True` thread team only oversubscribes the machine. Each worker still costs ~145 MB (Numba + LLVM import, ~40 MB of it render), so a 32-worker pool is ~4.5 GB – size the pool to the sections a render actually uses before raising it.
  **Numba pitfall:** the DD kernels must be compiled **without `fastmath=True`** – TwoSum/TwoProd rely on strict IEEE round-to-nearest; fast-math contraction silently destroys the error term. The same applies to any new kernel touching DD arithmetic.

## Development guidance
- **Adding a new color theme**: Implement a function with signature `(iterations, max_iterations, colors)` similar to existing ones and register it in `theme_functions`. Update the `self.themes` list if you want it selectable from the UI. The high-resolution themes (Rainbow2, Rainbow3, Rainbow4) build HSV palettes via `_build_hsv_palette()` with 1024/8192 hue steps.
- **Extending the calculator**: If performance needs improvement, consider splitting the image into more sections or adjusting `num_sections = 2 ** int(np.floor(np.log2(self.num_cores)))` to better match your CPU count.
- **Debugging**: Run with Python’s debugger:
  ```bash
  python -m pdb Fractal_Frontier.py
  ```
  Use breakpoints in `MandelbrotViewer.draw_mandelbrot()` or the JIT function as needed.
- **Bookmarks**: The current implementation stores bookmarks in `bookmarks.json`. Modify the file directly if you wish to pre‑populate views. The application now supports saving and loading bookmarks for different fractal types (Mandelbrot, Julia, Fatou) with their associated parameters.

## Repository conventions
- **Commit messages** follow Conventional Commits: `type(scope): summary` —
  e.g. `perf(pan): …`, `fix(zoom): …`, `feat(ux): …`, `docs(release): …`.
  Common types: `feat`, `fix`, `perf`, `refactor`, `test`, `build`, `chore`, `style`, `docs`.
- **Version bumps**: the app version lives in two places — `self.app_name`
  in `Fractal_Frontier.py` (e.g. `"Fractal Frontier V1.3"`) and the
  `**Version:**` line in `README.md`. Bump both in the same `docs(release):` commit.
- **Formatting**: black + isort (`make format`); keep lines ≤ 88 so flake8
  and black agree. Main-code files are kept flake8-clean (`tests/` is excluded).
- **Tests**: plain pytest under `tests/`. Long-running tests carry
  `@pytest.mark.timeout(n)` (pytest-timeout plugin). GPU tests are guarded
  with `@pytest.mark.skipif(not GPU_AVAILABLE, …)` so the default test run
  never requires CUDA hardware.
- **Public API**: the root `__init__.py` re-exports `MandelbrotCalculator`,
  the JIT section kernels, and `apply_color_theme`. Keep it in sync when
  adding public entry points.
- **Dev tools**: flake8, pytest, pytest-timeout, black, isort, and autopep8
  are pinned in `requirements.txt` and installed by `make install`.

## Licensing and usage notes
The repository is licensed under the MIT License. All code may be used, modified, and redistributed freely.
