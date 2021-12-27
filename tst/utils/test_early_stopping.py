import pytest

from src.utils.early_stopping import EarlyStopping


def test_early_stopper_stops_at_max_epochs():
    early_stopper = EarlyStopping(patience=1, max_epochs=100)
    stopped_at = None
    for epoch, accuracy in enumerate(range(1000)):
        early_stopper.update(-accuracy)
        if early_stopper.abort:
            stopped_at = early_stopper.curr_step
            break

    assert stopped_at == 101


def test_early_stopper_never_stops_before_min_patience_reached():
    early_stopper = EarlyStopping(patience=20, max_epochs=100)
    stopped_at = None
    for epoch, accuracy in enumerate(range(1000)):
        early_stopper.update(42.0 + accuracy)
        if early_stopper.abort:
            stopped_at = early_stopper.curr_step
            break
    assert stopped_at == 22


def test_early_stopper_stops_at_no_patience_left():
    early_stopper = EarlyStopping(patience=34, max_epochs=1000)
    stopped_at = None
    for epoch, accuracy in enumerate(range(1000)):
        if epoch < 500:
            early_stopper.update(42.0 - accuracy)
        else:
            early_stopper.update(42.0 + accuracy)
        if early_stopper.abort:
            stopped_at = early_stopper.curr_step
            break
    assert stopped_at == 501
