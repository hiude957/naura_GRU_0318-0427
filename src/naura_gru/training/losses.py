"""Masked loss and sample-weight utilities."""


def masked_sensor_loss(_pred, _target, _loss_mask, _sample_weight=None):
    """Compute loss only on valid sensor target dimensions."""
    raise NotImplementedError("Implement masked sensor loss here.")

