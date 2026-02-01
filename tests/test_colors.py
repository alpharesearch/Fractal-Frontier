import os
import sys

import numpy as np

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from Fractal_Frontier import apply_color_theme


def test_apply_color_theme_shape():
    iterations = np.zeros((10, 10), dtype=int)
    colors = apply_color_theme(iterations, max_iterations=100, theme="Default")
    assert colors.shape == (10, 10, 3)
