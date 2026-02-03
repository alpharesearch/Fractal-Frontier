import tkinter as tk
import pytest
from Fractal_Frontier import MandelbrotViewer


def _create_app():
    """Create a minimal Tkinter app that can instantiate MandelbrotViewer."""
    root = tk.Tk()
    # hide the window during testing
    root.withdraw()
    viewer = MandelbrotViewer(root)
    return root, viewer


@pytest.mark.timeout(5)
def test_zoom_persists_view_bounds_and_redraws():
    """Verify that a mouse‑wheel zoom updates bounds and triggers a redraw."""
    root, viewer = _create_app()

    # Capture current bounds
    original = (viewer.x_min, viewer.x_max, viewer.y_min, viewer.y_max)

    # Simulate a wheel event at the canvas centre
    class Event:
        type = "2"          # mousewheel / wheel scroll event in Tkinter
        x = viewer.width // 2
        y = viewer.height // 2

    from unittest.mock import patch

    # Perform zoom‑in (factor < 1) with mocked draw_mandelbrot to set image
    with patch.object(viewer, 'draw_mandelbrot', new=lambda: setattr(viewer.canvas, 'image', object())):
        viewer.zoom(Event(), zoom_factor=0.5)

    # After zoom, the overall width and height should decrease
    new_width = viewer.x_max - viewer.x_min
    new_height = viewer.y_max - viewer.y_min
    orig_width = original[1] - original[0]
    orig_height = original[3] - original[2]
    assert new_width < orig_width, "Zoom in should reduce width"
    assert new_height < orig_height, "Zoom in should reduce height"

    # The canvas should have been redrawn – verify image is not None
    assert viewer.canvas.image is not None

    root.destroy()
