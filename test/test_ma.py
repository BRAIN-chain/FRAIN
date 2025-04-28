from src.moving_average import MovingAverage


def test_get_current_average_without_new_value():
    ma = MovingAverage(3)
    ma.next(10)
    ma.next(20)
    # Do not add a new value, just get the current average
    current_avg = ma.current()
    expected_avg = (10 + 20) / 2  # As only two values have been added
    assert current_avg == expected_avg, "The current average did not match the expected value"

    # Test the behavior when no values have been added yet
    ma_empty = MovingAverage(3)
    current_avg_empty = ma_empty.current()
    assert current_avg_empty == 0, "Expected current average of empty MovingAverage to be 0"
