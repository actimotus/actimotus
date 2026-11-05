"""`Sens(placement='side')` turns a lateral thigh sensor to the front placement that ActiMotus expects.

The turn is 90 degrees about the long axis of the thigh: `x` stays, `y = -z`, `z = y`.
"""

from pathlib import Path

import numpy as np
import pytest

from actimotus import Sens

THIGH = Path(__file__).parent.parent / 'test_data' / 'thigh.bin'


@pytest.mark.parametrize('normalize', [True, False])
def test_side_turns_y_and_z_about_x(normalize):
    front = Sens(normalize=normalize).from_bin(THIGH)
    side = Sens(normalize=normalize, placement='side').from_bin(THIGH)

    assert side.index.equals(front.index)
    assert list(side.columns) == ['acc_x', 'acc_y', 'acc_z']
    assert side.dtypes.eq(np.float32).all()
    np.testing.assert_array_equal(side['acc_x'], front['acc_x'])
    np.testing.assert_array_equal(side['acc_y'], -front['acc_z'])
    np.testing.assert_array_equal(side['acc_z'], front['acc_y'])


def test_side_keeps_the_vector_length():
    front = Sens().from_bin(THIGH)
    side = Sens(placement='side').from_bin(THIGH)

    np.testing.assert_allclose(np.linalg.norm(side, axis=1), np.linalg.norm(front, axis=1), rtol=1e-6)


def test_front_is_the_default():
    np.testing.assert_array_equal(Sens().from_bin(THIGH), Sens(placement='front').from_bin(THIGH))


def test_unknown_placement_is_refused():
    with pytest.raises(ValueError, match='placement'):
        Sens(placement='back')  # type: ignore[arg-type]


def test_side_applies_to_a_buffer():
    buffer = THIGH.read_bytes()
    front = Sens().from_buffer(buffer)
    side = Sens(placement='side').from_buffer(buffer)

    np.testing.assert_array_equal(side['acc_y'], -front['acc_z'])
    np.testing.assert_array_equal(side['acc_z'], front['acc_y'])
