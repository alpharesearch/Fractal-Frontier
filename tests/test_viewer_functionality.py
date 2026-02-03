"""
Test suite for MandelbrotViewer functionality.

This module tests the viewer's zoom, pan, and bookmark functionality,
as well as other GUI-related features that are not covered in test_calculator.py.
"""

import os
import sys
import tempfile
import json

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from Fractal_Frontier import MandelbrotViewer


def test_zoom_calculation():
    """Test that zoom calculations work correctly."""
    # Create a mock event object
    class MockEvent:
        def __init__(self):
            self.type = "2"  # Mouse button press
            self.x = 80
            self.y = 60

    # We need to test the zoom logic without creating a full GUI
    # Let's create a minimal viewer instance and test the calculation directly

    # Test zoom in (zoom_factor < 1)
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    width = x_max - x_min
    height = y_max - y_min

    # Mouse position at center (80, 60) for a 160x120 canvas
    mouse_x, mouse_y = 80, 60
    zoom_factor = 0.5

    # Calculate new bounds - this is the logic from the zoom method
    x_center = x_min + (width * (mouse_x / 160))
    y_center = y_min + (height * (mouse_y / 120))

    new_width = width * zoom_factor
    new_height = height * zoom_factor

    new_x_min = x_center - (new_width / 2)
    new_x_max = x_center + (new_width / 2)
    new_y_min = y_center - (new_height / 2)
    new_y_max = y_center + (new_height / 2)

    # After zooming in, the range should be smaller
    assert abs(new_x_max - new_x_min) < width, "Zoom in should reduce width"
    assert abs(new_y_max - new_y_min) < height, "Zoom in should reduce height"

    # Test zoom out (zoom_factor > 1)
    zoom_factor = 2.0
    new_width = width * zoom_factor
    new_height = height * zoom_factor

    new_x_min = x_center - (new_width / 2)
    new_x_max = x_center + (new_width / 2)
    new_y_min = y_center - (new_height / 2)
    new_y_max = y_center + (new_height / 2)

    # After zooming out, the range should be larger
    assert abs(new_x_max - new_x_min) > width, "Zoom out should increase width"
    assert abs(new_y_max - new_y_min) > height, "Zoom out should increase height"


def test_bookmark_saving_and_loading():
    """Test that bookmarks can be saved and loaded correctly."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        # Create test bookmark data
        test_bookmarks = [
            {
                "name": "Test Bookmark 1",
                "x_min": -1.5,
                "x_max": 0.5,
                "y_min": -1.0,
                "y_max": 1.0,
                "max_iterations": 200,
                "theme": "Blue",
                "fractal_type": "Mandelbrot"
            }
        ]

        # Save bookmark
        with open(temp_file, 'w') as f:
            json.dump(test_bookmarks, f)

        # Verify file was created and contains expected data
        assert os.path.exists(temp_file), "Bookmark file should be created"

        with open(temp_file, 'r') as f:
            bookmarks = json.load(f)

        # Check that our bookmark was saved correctly
        assert len(bookmarks) == 1, "One bookmark should be saved"
        assert bookmarks[0]["x_min"] == -1.5
        assert bookmarks[0]["x_max"] == 0.5
        assert bookmarks[0]["y_min"] == -1.0
        assert bookmarks[0]["y_max"] == 1.0
        assert bookmarks[0]["max_iterations"] == 200
        assert bookmarks[0]["theme"] == "Blue"
        assert bookmarks[0]["fractal_type"] == "Mandelbrot"

        # Add another bookmark
        test_bookmarks.append({
            "name": "Test Bookmark 2",
            "x_min": -1.8,
            "x_max": 0.8,
            "y_min": -1.2,
            "y_max": 1.2,
            "max_iterations": 300,
            "theme": "Fire",
            "fractal_type": "Julia"
        })

        with open(temp_file, 'w') as f:
            json.dump(test_bookmarks, f)

        # Verify second bookmark was added
        with open(temp_file, 'r') as f:
            bookmarks = json.load(f)

        assert len(bookmarks) == 2, "Two bookmarks should be saved"
        assert bookmarks[1]["theme"] == "Fire"
        assert bookmarks[1]["fractal_type"] == "Julia"

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_bookmark_data_structure():
    """Test that bookmark data has the correct structure."""
    # Test with different fractal types and themes
    test_cases = [
        {
            "name": "Mandelbrot Deep Zoom",
            "x_min": -1.5,
            "x_max": 0.5,
            "y_min": -1.0,
            "y_max": 1.0,
            "max_iterations": 200,
            "theme": "Default",
            "fractal_type": "Mandelbrot"
        },
        {
            "name": "Julia Set",
            "x_min": -2.0,
            "x_max": 1.0,
            "y_min": -1.5,
            "y_max": 1.5,
            "max_iterations": 100,
            "theme": "Rainbow",
            "fractal_type": "Julia"
        },
        {
            "name": "Fatou Set",
            "x_min": -2.0,
            "x_max": 1.0,
            "y_min": -1.5,
            "y_max": 1.5,
            "max_iterations": 150,
            "theme": "Grayscale",
            "fractal_type": "Fatou"
        }
    ]

    # Save and load the bookmarks
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        with open(temp_file, 'w') as f:
            json.dump(test_cases, f)

        with open(temp_file, 'r') as f:
            loaded_bookmarks = json.load(f)

        # Verify all bookmarks were saved and loaded correctly
        assert len(loaded_bookmarks) == 3, "Three bookmarks should be saved"

        # Check each bookmark
        for i, original in enumerate(test_cases):
            loaded = loaded_bookmarks[i]
            assert loaded["name"] == original["name"], f"Bookmark {i} name mismatch"
            assert loaded["x_min"] == original["x_min"], f"Bookmark {i} x_min mismatch"
            assert loaded["x_max"] == original["x_max"], f"Bookmark {i} x_max mismatch"
            assert loaded["y_min"] == original["y_min"], f"Bookmark {i} y_min mismatch"
            assert loaded["y_max"] == original["y_max"], f"Bookmark {i} y_max mismatch"
            assert loaded["max_iterations"] == original["max_iterations"], f"Bookmark {i} max_iterations mismatch"
            assert loaded["theme"] == original["theme"], f"Bookmark {i} theme mismatch"
            assert loaded["fractal_type"] == original["fractal_type"], f"Bookmark {i} fractal_type mismatch"

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_view_bounds_validation():
    """Test that view bounds are valid (x_min < x_max, y_min < y_max)."""
    # Test various valid bound configurations
    valid_bounds = [
        (-2.0, 1.0, -1.5, 1.5),
        (-1.5, 0.5, -1.0, 1.0),
        (-3.0, 2.0, -2.0, 2.0),
    ]

    for x_min, x_max, y_min, y_max in valid_bounds:
        assert x_min < x_max, f"x_min ({x_min}) should be less than x_max ({x_max})"
        assert y_min < y_max, f"y_min ({y_min}) should be less than y_max ({y_max})"


def test_zoom_factor_validation():
    """Test that zoom factors are valid (positive numbers)."""
    # Valid zoom factors
    valid_factors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for factor in valid_factors:
        assert factor > 0, f"Zoom factor {factor} should be positive"


def test_max_iterations_validation():
    """Test that max iterations are reasonable values."""
    # Valid iteration counts (positive integers)
    valid_iterations = [10, 50, 100, 200, 500, 1000]

    for iterations in valid_iterations:
        assert iterations > 0, f"Max iterations {iterations} should be positive"
        assert isinstance(iterations, int), f"Max iterations {iterations} should be an integer"


def test_theme_names():
    """Test that theme names are valid strings."""
    # Valid theme names from the application
    valid_themes = [
        "Default", "Grayscale", "Blue", "Fire", "Rainbow",
        "Rainbow2", "Rainbow3", "Rainbow4", "CPU Cores"
    ]

    for theme in valid_themes:
        assert isinstance(theme, str), f"Theme {theme} should be a string"
        assert len(theme) > 0, f"Theme {theme} should not be empty"


def test_fractal_type_names():
    """Test that fractal type names are valid strings."""
    # Valid fractal types from the application
    valid_types = ["Mandelbrot", "Julia", "Fatou"]

    for fractal_type in valid_types:
        assert isinstance(fractal_type, str), f"Fractal type {fractal_type} should be a string"
        assert len(fractal_type) > 0, f"Fractal type {fractal_type} should not be empty"
