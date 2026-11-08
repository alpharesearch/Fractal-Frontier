# Fractal Frontier

**Version:** 1.3

## Overview
Fractal Frontier is a graphical application that allows users to explore the Mandelbrot set through an interactive interface. Users can zoom in and out, move the view, and apply different color themes to visualize the fractal patterns.

## Features
- Interactive zooming and panning of the Mandelbrot set.
- Multiple fractal types: Mandelbrot, Julia, and Fatou sets.
- Multiple color themes for visualizing the fractal, including high-resolution rainbow themes with finer color palettes.
- CPU Cores theme that randomly selects a color theme for each section.
- Bookmark functionality to save and load specific views with associated parameters.
- Efficient CPU calculations using multiprocessing and Numba JIT compilation.
- Optional GPU acceleration with Numba CUDA: float32 kernels for shallow zooms, float64 for deep zooms, and automatic CPU fallback if the GPU call fails.
- Responsive panning using a full-resolution raw-iteration tile cache, an asynchronously computed ring of adjacent tiles, and a bounded LRU history for quick direction reversals.
- Adaptive preview quality: full-resolution previews when the frame budget allows; half or quarter resolution only when a frame misses the budget.
- Live performance metrics in the title bar: render time to three decimal places and smoothed preview FPS.

## Project Structure
```
Fractal Frontier/
├── Fractal_Frontier.py           # Main application, viewer, and interaction logic
├── fractals.py                   # CPU/GPU calculator and backend selection
├── gpu_fractals.py               # Numba CUDA kernels and launch configuration
├── themes.py                     # Colour themes and palette helpers
├── requirements.txt              # Runtime and development dependencies
├── pytest.ini                    # Test configuration and warning policy
├── Makefile                      # Environment, test, lint, format, and run targets
├── README.md                     # Documentation for the project
├── CLAUDE.md                     # Architecture and development guidance
├── LICENSE                       # MIT License
├── .gitignore                    # Files and directories ignored by version control
└── tests/                        # Test suite directory
    ├── test_calculator.py        # Calculator and backend tests
    ├── test_colors.py            # Theme and colour tests
    ├── test_gpu.py               # GPU-specific tests
    ├── test_regressions.py       # Regression tests for the viewer and calculator
    ├── test_viewer_functionality.py
    └── test_viewer_integration.py
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/alpharesearch/Fractal-Frontier.git
   cd Fractal-Frontier
   ```

2. Create the conda environment and install dependencies:
   ```
   make conda-env
   ```

   If you already have the environment active, `make install` installs the dependencies into the current interpreter.

## Usage
To run the application, execute the following command:
```
make run
```

To run the test suite, execute:
```
make test
```

To run a specific test file, execute:
```
pytest tests/test_regressions.py
```

To lint the code, execute:
```
make lint
```

To format the code, execute:
```
make format
```

To clean build artifacts, execute:
```
make clean
```

To see all available targets, execute:
```
make help
```

## Rendering and performance

- **CPU backend:** Numba JIT kernels compute vertical sections in parallel, and a multiprocessing pool renders full frames across CPU cores.
- **GPU backend:** Numba CUDA kernels use float32 for shallow zooms and float64 where extra precision is needed. A runtime GPU failure falls back to CPU and updates the checkbox.
- **Pan tile cache:** every settled full render returns raw iteration counts. The viewer seeds full-resolution 256×256 tiles and the worker asynchronously computes a ring of three adjacent tiles. Up to 384 recently used tiles are retained in an LRU history, so a quick left/right reversal can reuse them. Active tiles are never evicted.
- **Preview quality:** previews start at full resolution. If a measured frame misses the frame budget, the viewer steps down to half or quarter resolution, then recovers after several consecutive cheap frames. The current target is 24 fps.
- **Title metrics:** the title bar shows settled render time with three decimals and a smoothed preview FPS based on displayed preview-frame intervals.

The tile grid is invalidated on zoom, fractal-type, iteration-count, or canvas-size changes. The LRU history is retained during a drag; a settled full render re-anchors the grid.

Numba may print `NumbaPerformanceWarning` for very small preview images. The 256×256 tiles launch 256 GPU blocks, above Numba's 128-block heuristic, so tile launches no longer trigger it. The test configuration ignores these warnings for the intentionally small probe images.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.