# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Generation-scoped records for the native forward split-stage API.

The record is deliberately an orchestration object, not a physics fallback.
It carries borrowed native outputs from one stage to the next and rejects a
record after state mutation, replacement, or an out-of-order stage call.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import numbers


class ForwardStage(IntEnum):
  NONE = 0
  POS = 1
  VEL = 2
  ACT = 3
  ACC = 4
  CONSTRAINT = 5


@dataclass
class ForwardStageRecord:
  """Borrowed result bundle for one coherent forward evaluation."""

  owner: object
  generation: int
  epoch: int
  token: int
  qpos: object
  mocap_pos: object
  mocap_quat: object
  input_tensors: dict = field(default_factory=dict)
  input_versions: dict = field(default_factory=dict)
  stage: ForwardStage = ForwardStage.NONE
  values: dict = field(default_factory=dict)


class ForwardStageCoordinator:
  """Enforce POS→VEL→ACT→ACC→CONSTRAINT ownership and invalidation.

  Callers execute the actual native physics kernels and publish each result
  with :meth:`publish`.  This class never reads tensor values or copies device
  storage.  A new POS call replaces the previous record, and state generation
  changes invalidate it before any later-stage consumer can reuse cached data.
  """

  def __init__(self, owner):
    self._owner = owner
    self._epoch = 0
    self._next_token = 1
    self._record = None

  @property
  def epoch(self):
    return self._epoch

  def invalidate(self):
    """Invalidate all borrowed stage views after reset, restore, or mutation."""
    self._epoch += 1
    self._record = None

  def begin(self, *, generation, qpos, mocap_pos=None, mocap_quat=None,
            position=None):
    """Start a fresh position evaluation and optionally publish its output."""
    self.invalidate()
    record = ForwardStageRecord(
        owner=self._owner, generation=int(generation), epoch=self._epoch,
        token=self._next_token, qpos=qpos, mocap_pos=mocap_pos,
        mocap_quat=mocap_quat)
    self._record = record
    for name, value in (("qpos", qpos), ("mocap_pos", mocap_pos),
                        ("mocap_quat", mocap_quat)):
      if value is not None:
        self.capture_input(record, name, value)
    self._next_token += 1
    self._record = record
    if position is not None:
      self.publish(record, ForwardStage.POS, position)
    return record

  def validate(self, record, *, generation, minimum=ForwardStage.POS):
    """Return the live record or raise for stale/foreign/out-of-order use."""
    current = self._record
    if (not isinstance(record, ForwardStageRecord)
        or current is not record or record.owner is not self._owner
        or record.epoch != self._epoch
        or record.generation != int(generation)):
      raise ValueError("forward-stage record is stale or belongs to another simulation")
    required = ForwardStage(minimum)
    if record.stage < required:
      raise ValueError(
          f"forward-stage record is at {record.stage.name}, needs {required.name}")
    self._validate_captured_inputs(record, required)
    return record

  @staticmethod
  def _version(value):
    """Return a tensor mutation counter, or None when unavailable."""
    try:
      version = value._version
    except (AttributeError, RuntimeError, TypeError):
      return None
    return version if isinstance(version, numbers.Integral) else None

  def capture_input(self, record, name, value):
    """Bind a stage input to its identity and mutation counter."""
    current = self._record
    if current is not record or record.epoch != self._epoch:
      raise ValueError("cannot bind an input to a stale forward-stage record")
    record.input_tensors[name] = value
    record.input_versions[name] = self._version(value)

  def validate_input(self, record, name, value=None):
    """Reject replacement or in-place mutation of cached stage inputs."""
    captured = record.input_tensors.get(name)
    if captured is None:
      raise ValueError(f"forward-stage input {name} was not captured")
    if name in ("qpos", "mocap_pos", "mocap_quat"):
      if getattr(record, name) is not captured:
        raise ValueError(f"cached {name} input identity was replaced")
    elif name == "qvel" and record.stage >= ForwardStage.VEL:
      velocity = record.values.get(ForwardStage.VEL, {})
      if velocity.get("qvel") is not captured:
        raise ValueError("cached qvel input identity was replaced")
    if value is not None and value is not captured:
      raise ValueError(f"{name} does not match the cached stage input")
    expected = record.input_versions.get(name)
    actual = self._version(captured)
    if expected is None or actual is None:
      raise ValueError(
          f"cached {name} reuse requires a tensor mutation counter")
    if actual != expected:
      raise ValueError(f"cached {name} was mutated after its stage")

  def _validate_captured_inputs(self, record, required):
    # POS storage is borrowed through every later stage. VEL storage becomes
    # borrowed once that stage is published.
    if required >= ForwardStage.POS:
      for name in ("qpos", "mocap_pos", "mocap_quat"):
        if name in record.input_tensors:
          self.validate_input(record, name)
    if required >= ForwardStage.VEL and "qvel" in record.input_tensors:
      self.validate_input(record, "qvel")

  def current(self, *, generation, minimum=ForwardStage.POS):
    """Return the latest prepared record if it still satisfies this request."""
    if self._record is None:
      raise ValueError("no forward-stage record has been prepared")
    return self.validate(self._record, generation=generation, minimum=minimum)

  def publish(self, record, stage, values, *, generation=None):
    """Publish/replace a stage and invalidate every later cached output.

    MuJoCo callers may evaluate a stage repeatedly at one state (for example,
    ACT with two control overrides, then ACC). Replacing stage ``s`` retains
    the prefix before ``s`` and drops all values derived from ``s`` onward.
    """
    stage = ForwardStage(stage)
    prefix = ForwardStage(max(int(ForwardStage.NONE), int(stage) - 1))
    current = self.validate(
        record, generation=(record.generation if generation is None else generation),
        minimum=ForwardStage.NONE)
    if stage > current.stage + 1:
      raise ValueError(
          f"forward stage cannot skip from {current.stage.name} to {stage.name}")
    if prefix > ForwardStage.NONE:
      current = self.validate(
          record,
          generation=(record.generation if generation is None else generation),
          minimum=prefix)
    if not isinstance(values, dict):
      raise TypeError("forward-stage values must be a dictionary")
    # Bind a refreshed VEL input only after every validation that can reject
    # publication. This lets POS be reused with a new qvel while ensuring a
    # failed/out-of-order publish leaves the live record unchanged.
    if stage == ForwardStage.VEL:
      qvel = values.get("qvel")
      if qvel is None:
        current.input_tensors.pop("qvel", None)
        current.input_versions.pop("qvel", None)
      else:
        self.capture_input(current, "qvel", qvel)
    for later in tuple(current.values):
      if later >= stage:
        del current.values[later]
    current.values[stage] = values
    current.stage = stage
    return current

  def consume(self, record, stage, *, generation):
    """Validate and return a previously published stage result."""
    current = self.validate(record, generation=generation, minimum=stage)
    try:
      return current.values[ForwardStage(stage)]
    except KeyError as exc:
      raise ValueError(f"forward stage {ForwardStage(stage).name} has no result") from exc
