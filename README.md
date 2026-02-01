# Fractal Frontier

## Overview
Fractal Frontier is a graphical application that allows users to explore the Mandelbrot set through an interactive interface. Users can zoom in and out, move the view, and apply different color themes to visualize the fractal patterns.

## Features
- Interactive zooming and panning of the Mandelbrot set.
- Multiple fractal types: Mandelbrot, Julia, and Fatou sets.
- Multiple color themes for visualizing the fractal, including high-resolution themes with clipping to avoid overflow.
- CPU Cores theme that randomly selects a color theme for each section.
- Bookmark functionality to save and load specific views with associated parameters.
- Efficient calculations using multiprocessing and JIT compilation with support for different fractal types.

## Project Structure
```
Fractal Frontier
├── Fractal_Frontier.py           # The main application
├── requirements.txt              # Lists the dependencies required for the project.
├── README.md                     # Documentation for the project.
├── LICENSE                       # MIT License
├── .gitignore                    # Specifies files and directories to ignore in version control.
└── tests/                        # Test suite directory
    └── test_calculator.py        # Unit tests for the calculator and theme system
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/alpharesearch/Fractal-Frontier.git
   cd Fractal Frontier
   ```

2. Install the required dependencies:
   ```
   make install
   ```

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
pytest tests/test_calculator.py
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

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.