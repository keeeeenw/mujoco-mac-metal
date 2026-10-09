# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Integrated admission must pass the original source dt to native stages."""
import mujoco
import numpy as np
import pytest
from mujoco_metal.stepping import validate_stepping_profile

@pytest.mark.parametrize('integrator,profile',[
 ('Euler','integrated_euler_v1'),('Euler','integrated_scalable_v1'),
 ('RK4','integrated_rk4_v1'),('implicit','integrated_implicit_v1'),
 ('implicitfast','integrated_implicitfast_v1')])
@pytest.mark.parametrize('override',[None,.001234567890123])
def test_integrated_profiles_keep_source_timestep(integrator,profile,override):
 m=mujoco.MjModel.from_xml_string(f'<mujoco><option timestep=".0005" integrator="{integrator}"><flag contact="disable"/></option><worldbody><body><joint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
 expected=m.opt.timestep if override is None else override
 p=validate_stepping_profile(m,timestep=override,profile=profile)
 assert p.timestep==expected
 assert p.timestep!=float(np.float32(expected))
 assert p.execution_plan.timestep==expected
