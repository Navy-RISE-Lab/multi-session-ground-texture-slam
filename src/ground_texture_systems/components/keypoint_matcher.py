"""Construct a KeypointMatcher component."""

from dataclasses import dataclass

import ground_texture_slam


@dataclass
class KeypointMatcherOptions:
    """Options for the KeypointMatcher component."""

    match_threshold: float = 0.6
    wta_k: int = 2


def create_keypoint_matcher(
    options: KeypointMatcherOptions,
) -> ground_texture_slam.KeypointMatcher:
    """Create a KeypointMatcher object.

    Parameters
    ----------
    options
        Options for this component.

    Returns
    -------
        The constructed object.
    """
    keypoint_matcher_options = ground_texture_slam.KeypointMatcher.Options()
    keypoint_matcher_options.match_threshold = options.match_threshold
    keypoint_matcher_options.WTA_K = options.wta_k
    return ground_texture_slam.KeypointMatcher(keypoint_matcher_options)
