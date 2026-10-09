# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned contact-axis records and their retained regularizer sidecar.

The original six seven-word axes remain the public contact-data view. Internal
records append six regularizer high/low pairs and five friction low words.
"""
AXIS_WORDS = 7
AXIS_COUNT = 6
AXIS_DATA_WORDS = AXIS_WORDS * AXIS_COUNT
REGULARIZER_HIGH_OFFSET = 42
REGULARIZER_LOW_OFFSET = 48
FRICTION_LOW_OFFSET = 54
RECORD_WORDS = 60


def axis_data(records, batch, slots):
  """Borrow the unchanged axis fields from the complete owned slot records."""
  owned = records[:batch * slots * RECORD_WORDS].reshape(batch, slots, RECORD_WORDS)
  return owned[..., :AXIS_DATA_WORDS].reshape(batch, slots, AXIS_COUNT, AXIS_WORDS)
