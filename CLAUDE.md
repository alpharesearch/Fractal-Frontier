# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick start commands
- **Install dependencies**
  ```bash
  make install
  ```
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
- **Linting** – No lint configuration is provided; however, you may use flake8 manually:
  ```bash
  pip install flake8
  flake8 Fractal_Frontier.py
  ```
- **Formatting** – The project uses black and isort for code formatting. Run with:
  ```bash
  make format
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
├── Fractal_Frontier.py      # Main application code (GUI + Mandelbrot calculations)
├── requirements.txt         # Pillow, numpy, numba
├── README.md                # High‑level documentation
├── LICENSE                  # MIT license
└── tests/                   # Test suite directory
    └── test_calculator.py   # Unit tests for the calculator and theme system
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
- **GPU backend** – `fractals.py` sets `GPU_AVAILABLE` only when a real CUDA device is present (importing numba-cuda alone is not enough). `MandelbrotCalculator._compute_iterations()` falls back to the CPU JIT path if a GPU call fails at runtime, and the viewer syncs its GPU checkbox from the calculator state.

## Development guidance
- **Adding a new color theme**: Implement a function with signature `(iterations, max_iterations, colors)` similar to existing ones and register it in `theme_functions`. Update the `self.themes` list if you want it selectable from the UI. The high-resolution themes (Rainbow2, Rainbow3, Rainbow4) build HSV palettes via `_build_hsv_palette()` with 1024/8192 hue steps.
- **Extending the calculator**: If performance needs improvement, consider splitting the image into more sections or adjusting `num_sections = 2 ** int(np.floor(np.log2(self.num_cores)))` to better match your CPU count.
- **Debugging**: Run with Python’s debugger:
  ```bash
  python -m pdb Fractal_Frontier.py
  ```
  Use breakpoints in `MandelbrotViewer.draw_mandelbrot()` or the JIT function as needed.
- **Bookmarks**: The current implementation stores bookmarks in `bookmarks.json`. Modify the file directly if you wish to pre‑populate views. The application now supports saving and loading bookmarks for different fractal types (Mandelbrot, Julia, Fatou) with their associated parameters.

## Licensing and usage notes
The repository is licensed under the MIT License. All code may be used, modified, and redistributed freely.
