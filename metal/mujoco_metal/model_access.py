# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Internal prepared-model access, without copying the public inspection view."""


def prepared_model(sim):
  try:
    return sim._mjmodel
  except AttributeError:
    # Host adapters used by independent query tests expose only their model.
    return sim.model
