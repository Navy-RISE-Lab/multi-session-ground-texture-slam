"""Define a standard frame measurement."""

from dataclasses import dataclass

import numpy as np
from gtsam import Pose2
from numpy.typing import NDArray


@dataclass
class Measurement:
    """A class for individual frame measurements."""

    index: int
    session_id: int
    measurement_id: int
    image: NDArray[np.number]
    actual_pose: Pose2
