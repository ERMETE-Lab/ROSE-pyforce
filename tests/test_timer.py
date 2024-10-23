from pyforce.tools.timer import Timer, TimerError

def test_timer_start_stop():
    timer = Timer()
    timer.start()
    elapsed_time = timer.stop()
    assert elapsed_time >= 0