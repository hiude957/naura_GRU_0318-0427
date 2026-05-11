"""Project accuracy metrics for normalized sensor predictions."""


def normalized_accuracy(pred, true, valid_mask, near_zero_threshold=0.005, relative_threshold=0.20):
    """Compute project accuracy for normalized sensor values.

    For abs(true) > threshold, relative error must be below relative_threshold.
    For abs(true) <= threshold, abs(pred) must be below threshold.
    """
    raise NotImplementedError("Implement vectorized normalized accuracy here.")

