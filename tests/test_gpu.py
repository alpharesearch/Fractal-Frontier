"""Unit tests for GPU-accelerated fractal calculations."""

import numpy as np
import pytest

from fractals import (GPU_AVAILABLE, MandelbrotCalculator,
                      mandelbrot_section_jit)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUCalculator:
    """Tests for the GPU calculator module."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.width = 160
        self.height = 120
        self.x_min = -2.0
        self.x_max = 1.0
        self.y_min = -1.5
        self.y_max = 1.5
        self.max_iterations = 100
        self.calculator = MandelbrotCalculator()

    def test_gpu_availability(self):
        """Verify GPU is available when tests run."""
        assert GPU_AVAILABLE is True

    def test_mandelbrot_gpu_output_shape(self):
        """GPU Mandelbrot should return correct shape array."""
        self.calculator.set_use_gpu(True)
        result = self.calculator.calculate_full(
            "mandelbrot",
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
            "Default",
        )
        assert result.shape == (self.height, self.width, 3)  # RGB

    def test_mandelbrot_gpu_values_in_range(self):
        """GPU Mandelbrot iteration counts should be in valid range."""
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        iterations = gpu.calculate_mandelbrot_gpu(
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
        )
        assert iterations.min() >= 0
        assert iterations.max() <= self.max_iterations

    def test_mandelbrot_gpu_matches_cpu(self):
        """GPU and CPU Mandelbrot should produce nearly identical results."""
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        gpu_result = gpu.calculate_mandelbrot_gpu(
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
        )
        cpu_result = mandelbrot_section_jit(
            0,
            self.width,
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
        )
        # Allow small differences due to floating-point boundary cases
        diff = gpu_result != cpu_result
        diff_pct = np.sum(diff) / gpu_result.size
        assert diff_pct < 0.001  # Less than 0.1% pixel difference

    def test_mandelbrot_float32_matches_float64(self):
        """The float32 kernel should match the float64 kernel closely at the
        classic view (where float32 is selected by the viewer)."""
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        args = (
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
        )
        result32 = gpu.calculate_mandelbrot_gpu(*args, precision="float32")
        result64 = gpu.calculate_mandelbrot_gpu(*args, precision="float64")
        diff_pct = np.mean(result32 != result64)
        assert diff_pct < 0.01, f"float32/float64 mismatch: {diff_pct:.3%}"

    def test_julia_gpu_output_shape(self):
        """GPU Julia should return correct shape array."""
        self.calculator.set_use_gpu(True)
        julia_c = complex(-0.7, 0.27015)
        result = self.calculator.calculate_full(
            "julia",
            self.width,
            self.height,
            -1.5,
            1.5,
            -1.5,
            1.5,
            self.max_iterations,
            "Default",
            c=julia_c,
        )
        assert result.shape == (self.height, self.width, 3)

    def test_julia_gpu_values_in_range(self):
        """GPU Julia iteration counts should be in valid range."""
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        julia_c = complex(-0.7, 0.27015)
        iterations = gpu.calculate_julia_gpu(
            self.width, self.height, -1.5, 1.5, -1.5, 1.5, self.max_iterations, julia_c
        )
        assert iterations.min() >= 0
        assert iterations.max() <= self.max_iterations

    def test_julia_gpu_matches_cpu(self):
        """GPU and CPU Julia should produce nearly identical results."""
        from fractals import julia_section_jit
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        julia_c = complex(-0.7, 0.27015)
        gpu_result = gpu.calculate_julia_gpu(
            self.width, self.height, -1.5, 1.5, -1.5, 1.5, self.max_iterations, julia_c
        )
        cpu_result = julia_section_jit(
            0,
            self.width,
            self.width,
            self.height,
            -1.5,
            1.5,
            -1.5,
            1.5,
            self.max_iterations,
            julia_c,
        )
        diff = gpu_result != cpu_result
        diff_pct = np.sum(diff) / gpu_result.size
        assert diff_pct < 0.001

    def test_fatou_gpu_output_shape(self):
        """GPU Fatou should return correct shape array."""
        self.calculator.set_use_gpu(True)
        result = self.calculator.calculate_full(
            "fatou",
            self.width,
            self.height,
            -2.0,
            2.0,
            -2.0,
            2.0,
            self.max_iterations,
            "Default",
        )
        assert result.shape == (self.height, self.width, 3)

    def test_fatou_gpu_values_in_range(self):
        """GPU Fatou iteration counts should be in valid range."""
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        iterations = gpu.calculate_fatou_gpu(
            self.width, self.height, -2.0, 2.0, -2.0, 2.0, self.max_iterations
        )
        assert iterations.min() >= 0
        assert iterations.max() <= self.max_iterations

    def test_fatou_gpu_matches_cpu(self):
        """GPU and CPU Fatou should produce nearly identical results."""
        from fractals import fatou_section_jit
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        gpu_result = gpu.calculate_fatou_gpu(
            self.width, self.height, -2.0, 2.0, -2.0, 2.0, self.max_iterations
        )
        cpu_result = fatou_section_jit(
            0,
            self.width,
            self.width,
            self.height,
            -2.0,
            2.0,
            -2.0,
            2.0,
            self.max_iterations,
        )
        diff = gpu_result != cpu_result
        diff_pct = np.sum(diff) / gpu_result.size
        assert diff_pct < 0.001

    def test_gpu_toggle(self):
        """Toggling GPU on/off should change behavior."""
        # GPU mode
        self.calculator.set_use_gpu(True)
        gpu_result = self.calculator.calculate_full(
            "mandelbrot",
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
            "Default",
        )

        # CPU mode
        self.calculator.set_use_gpu(False)
        cpu_result = self.calculator.calculate_full(
            "mandelbrot",
            self.width,
            self.height,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
            "Default",
        )

        # Both should produce valid RGB arrays
        assert gpu_result.shape == cpu_result.shape
        # Results should be similar (may have small floating-point diffs)
        diff = np.abs(gpu_result.astype(float) - cpu_result.astype(float))
        assert np.mean(diff) < 5.0  # Average color difference is small

    def test_higher_resolution(self):
        """GPU should handle higher resolutions correctly."""
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        result = gpu.calculate_mandelbrot_gpu(
            800,
            600,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.max_iterations,
        )
        assert result.shape == (600, 800)

    def test_zoomed_view(self):
        """GPU should produce correct results for zoomed-in views."""
        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()
        # Region on the Mandelbrot boundary (not inside cardioid/bulb)
        zoom_x_min = -1.0
        zoom_x_max = -0.5
        zoom_y_min = 0.2
        zoom_y_max = 0.7

        gpu_result = gpu.calculate_mandelbrot_gpu(
            self.width, self.height, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max, 500
        )
        assert gpu_result.shape == (self.height, self.width)
        # Should have variation in iteration counts (boundary region)
        unique_values = np.unique(gpu_result)
        assert len(unique_values) > 10  # At least some variety in the fractal boundary

    def test_performance(self):
        """GPU should be significantly faster than CPU for a compute-bound
        image.

        The classic view is a poor benchmark: the periodicity checks and
        fast escaping make most pixels 1-iteration work, which a modern
        many-core CPU handles almost as well as a GPU (1.7x measured on a
        Ryzen 9950X). A deep boundary region (seahorse valley) forces most
        pixels to run the full iteration count — compute-bound work where
        the GPU's thousands of threads dominate.
        """
        import time

        from gpu_fractals import GPUCalculator

        gpu = MandelbrotCalculator.get_gpu_calculator() or GPUCalculator()

        # Compute-heavy: seahorse valley boundary, high iteration count.
        w, h, iters = 1920, 1080, 2000
        bx_min, bx_max, by_min, by_max = -0.75, -0.73, 0.12, 0.14

        # Warm up GPU (first call compiles kernel)
        _ = gpu.calculate_mandelbrot_gpu(w, h, bx_min, bx_max, by_min, by_max, iters)

        # Warm up CPU JIT
        _ = mandelbrot_section_jit(0, w, w, h, bx_min, bx_max, by_min, by_max, iters)

        # Benchmark GPU - use perf_counter for high-resolution timing
        t0 = time.perf_counter()
        for _ in range(20):
            gpu.calculate_mandelbrot_gpu(w, h, bx_min, bx_max, by_min, by_max, iters)
        gpu_time = (time.perf_counter() - t0) / 20

        # Benchmark CPU
        t0 = time.perf_counter()
        for _ in range(20):
            mandelbrot_section_jit(0, w, w, h, bx_min, bx_max, by_min, by_max, iters)
        cpu_time = (time.perf_counter() - t0) / 20

        speedup = cpu_time / gpu_time
        print(
            f"\nGPU: {gpu_time*1000:.1f}ms | CPU: {cpu_time*1000:.1f}ms | Speedup: {speedup:.1f}x"
        )
        # GPU is faster but transfer overhead limits speedup for single-frame benchmarks
        assert speedup > 2.0, f"GPU should be at least 2x faster, got {speedup:.1f}x"


def test_invalid_precision_rejected():
    """Only the two supported GPU precisions are accepted."""
    from gpu_fractals import GPUCalculator

    with pytest.raises(ValueError):
        GPUCalculator._validate_precision("float16")

    for precision in ("float32", "float64"):
        GPUCalculator._validate_precision(precision)  # must not raise


@pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires GPU to be unavailable")
class TestGPUNotAvailable:
    """Tests for graceful fallback when GPU is not available."""

    def test_gpu_available_is_false(self):
        """GPU_AVAILABLE should be False when no GPU."""
        assert GPU_AVAILABLE is False

    def test_set_use_gpu_fallback_warning(self):
        """Setting GPU to True when unavailable should warn."""
        calc = MandelbrotCalculator()
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        calc.set_use_gpu(True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "WARNING" in output or "not available" in output.lower()
