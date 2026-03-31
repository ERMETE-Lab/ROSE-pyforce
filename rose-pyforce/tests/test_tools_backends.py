import numpy as np
import pyvista as pv
import pytest
import time
from pyforce.tools.backends import IntegralCalculator, LoopProgress, Timer, TimerError

# -----------------------
# Tests for IntegralCalculator
# -----------------------

@pytest.fixture(params=[2,3])
def simple_grid(request):
    """Create a simple uniform grid in 2D or 3D."""
    gdim = request.param
    if gdim == 2:
        # Create a 2D square mesh (1x1 area)
        grid = pv.Plane(i_resolution=1, j_resolution=1)
    else:
        # Create a sphere
        grid = pv.examples.load_hexbeam()
    return grid, gdim

@pytest.fixture
def calc(simple_grid):
    grid, gdim = simple_grid
    return IntegralCalculator(grid, gdim=gdim)

def test_cell_sizes(calc):
    """Check that cell sizes are positive and match number of cells."""
    assert np.all(calc.cell_sizes > 0)
    assert calc.cell_sizes.shape[0] == calc.n_cells, f"Expected {calc.n_cells} cells, got {calc.cell_sizes.shape[0]}"

def test_check_input_scalar_point_data(calc):
    """Check scalar field defined at points is converted to cell data."""
    u = np.ones(calc.n_points)
    u_cell = calc.check_input(u)
    assert u_cell.shape[0] == calc.n_cells

def test_check_input_scalar_cell_data(calc):
    """Check scalar field defined at cells is returned unchanged."""
    u = np.ones(calc.n_cells)
    u_cell = calc.check_input(u)
    assert np.allclose(u_cell, u)


def test_check_input_vector_field(calc):
    """Check vector field reshaping for both point-based and cell-based data."""
    u = np.ones(calc.n_points * calc.gdim)
    u_r = calc.check_input(u)
    assert u_r.shape == (calc.n_points, calc.gdim)

    u = np.ones(calc.n_cells * calc.gdim)
    u_r = calc.check_input(u)
    assert u_r.shape == (calc.n_cells, calc.gdim)


def test_integral_constant_field(calc):
    """Integral of constant scalar field should equal value * total cell measure."""
    u = np.ones(calc.n_cells)
    integral = calc.integral(u)
    expected = np.sum(calc.cell_sizes)
    assert np.isclose(integral, expected)


def test_average_constant_field(calc):
    """Average of constant field should equal the constant itself."""
    u = np.ones(calc.n_cells)
    ave = calc.average(u)
    assert np.isclose(ave, 1.0), f"Expected average 1.0, got {ave}"


def test_L1_norm(calc):
    """L1 norm of a constant field."""
    u = np.ones(calc.n_cells)
    L1 = calc.L1_norm(u)
    expected = np.sum(calc.cell_sizes)
    assert np.isclose(L1, expected)


def test_L2_inner_product(calc):
    """Inner product of ones with itself should equal total measure."""
    u = np.ones(calc.n_cells)
    v = np.ones(calc.n_cells)
    prod = calc.L2_inner_product(u, v)
    expected = np.sum(calc.cell_sizes)
    assert np.isclose(prod, expected)


def test_L2_norm(calc):
    """L2 norm should be sqrt(total measure) for a field of ones."""
    u = np.ones(calc.n_cells)
    norm = calc.L2_norm(u)
    expected = np.sqrt(np.sum(calc.cell_sizes))
    assert np.isclose(norm, expected)


def test_vector_field_integral(calc):
    """Check integral of constant vector field."""
    u = np.ones(calc.n_cells * calc.gdim)
    u = calc.check_input(u)
    res = calc.integral(u)
    expected = np.ones(calc.gdim) * np.sum(calc.cell_sizes)
    assert np.allclose(res, expected)


def test_invalid_input_shape(calc):
    """Ensure invalid input raises ValueError."""
    u = np.ones(calc.n_cells + 5)
    with pytest.raises(ValueError):
        calc.check_input(u)

# -----------------------
# Tests for LoopProgress
# -----------------------

def test_loopprogress_initialization(capsys):
    """Test initialization prints the starting message."""
    lp = LoopProgress("ProgressTest", final=10)
    captured = capsys.readouterr()
    assert "ProgressTest:" in captured.out
    assert np.isclose(lp.instant, 0.0)
    assert lp.final == 10
    assert isinstance(lp.comp_times, list)


def test_loopprogress_update_numeric(capsys):
    """Test numeric updates produce correctly formatted output."""
    lp = LoopProgress("Loop", final=5)
    time.sleep(0.01)  # to ensure measurable time difference
    lp.update(1.0)
    captured = capsys.readouterr().out

    # Check output format
    assert "Loop:" in captured
    assert "1.000" in captured
    assert "/ 5.00" in captured
    assert "s/it" in captured


def test_loopprogress_update_percentage(capsys):
    """Test percentage-based update prints in %."""
    lp = LoopProgress("Percent", final=10)
    time.sleep(0.01)
    lp.update(5, percentage=True)
    captured = capsys.readouterr().out
    assert "%" in captured
    assert "Percent:" in captured
    assert "50.000" in captured  # halfway


def test_loopprogress_complete_linebreak(capsys):
    """When the loop completes, it should print with newline (no carriage return)."""
    lp = LoopProgress("Done", final=2)
    time.sleep(0.01)
    lp.update(1.0)
    lp.update(1.0)  # Should now reach final
    captured = capsys.readouterr().out
    assert "Done:" in captured
    # The final print should contain a newline rather than '\r'
    assert not captured.endswith("\r")


def test_loopprogress_computation_time_increases():
    """Ensure comp_times list grows and average time computed."""
    lp = LoopProgress("Timing", final=2)
    time.sleep(0.01)
    lp.update(1.0)
    n_before = len(lp.comp_times)
    time.sleep(0.01)
    lp.update(1.0)
    n_after = len(lp.comp_times)
    assert n_after == n_before + 1


# -----------------------
# Tests for Timer
# -----------------------

def test_timer_basic_usage():
    """Timer should measure elapsed CPU time."""
    t = Timer()
    t.start()
    _ = sum(range(10000))
    elapsed = t.stop()
    assert elapsed > 0
    assert t._start_time is None


def test_timer_double_start_raises():
    """Starting twice without stopping should raise."""
    t = Timer()
    t.start()
    with pytest.raises(TimerError):
        t.start()


def test_timer_stop_without_start_raises():
    """Stopping before starting should raise."""
    t = Timer()
    with pytest.raises(TimerError):
        t.stop()


def test_timer_restart_after_stop():
    """Timer can be reused after stopping."""
    t = Timer()
    t.start()
    t.stop()
    # Should not raise
    t.start()
    t.stop()