"""Retain the first failed step of the original soft/stiff native fixture."""
import json, os
from pathlib import Path
import numpy as np
import pytest
import mujoco
from mujoco_metal.simulation import MetalSimulation
from test_solver_completion_014 import PROFILE

@pytest.mark.gpu
def test_original_soft_stiff_first_failure_capture():
  torch=pytest.importorskip("torch")
  if os.environ.get("MUJOCO_METAL_RUN_GPU")!="1":pytest.skip("opt-in native qualification")
  xml=('<mujoco><option timestep="0.002" integrator="Euler" iterations="200" '
       'tolerance="1e-8" gravity="0 0 -9.81"/>'
       '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
       '<body pos="0 0 0.3"><joint name="soft" type="slide" axis="0 0 1" '
       'stiffness="20" damping="2"/>'
       '<geom name="ball" type="sphere" size="0.06"/></body>'
       '</worldbody></mujoco>')
  model=mujoco.MjModel.from_xml_string(xml)
  sim=MetalSimulation(model,batch_size=1,profile=PROFILE)
  qp=np.asarray(model.qpos0,np.float32).reshape(1,-1)
  sim.reset(qpos=qp,qvel=np.zeros((1,model.nv),np.float32))
  cpu=mujoco.MjData(model);cpu.qpos[:]=qp[0];mujoco.mj_forward(model,cpu)
  def array(t):return t.detach().cpu().numpy().tolist()
  records=[]; failed_calls=[]
  # Capture the failing solve before masked recovery overwrites borrowed outputs.
  owner=sim._coupled_constraints
  for method_name in ("run_device", "run_velocity_device"):
    original=getattr(owner,method_name)
    def observe(*args, _original=original, _name=method_name, **kwargs):
      inputs={"mass":array(args[1 if _name=="run_device" else 2]) if args[1 if _name=="run_device" else 2] is not None else None,
              "qfrc_hi":array(args[2 if _name=="run_device" else 3]),
              "qfrc_low":array(kwargs["qfrc_smooth_low"]) if kwargs.get("qfrc_smooth_low") is not None else None}
      result=_original(*args,**kwargs)
      status=array(result["status"])
      if any(status):
        failed_calls.append({"method":_name,"status":status,"inputs":inputs,
          "workspace":{k:array(owner._workspace[k]) for k in ("out_diagnostics","out_acc","out_force","workspace_debug")},
          "solver_dims":array(owner._constants["solver_dims"]),
          "world_mask":array(kwargs["world_mask"]) if kwargs.get("world_mask") is not None else None})
      return result
    setattr(owner,method_name,observe)
  for step in range(400):
    pre={"qpos":array(sim.state.qpos),"qvel":array(sim.state.qvel)}
    sim.step(1);mujoco.mj_step(model,cpu)
    status=array(sim.state.status)
    entry={"step":step+1,"pre":pre,"qpos":array(sim.state.qpos),"qvel":array(sim.state.qvel),"status":status,"cpu_qpos":cpu.qpos.tolist(),"cpu_qvel":cpu.qvel.tolist(),"cpu_qacc":cpu.qacc.tolist(),"cpu_niter":cpu.solver_niter.tolist()}
    if sim._coupled_constraints is not None:
      w=sim._coupled_constraints._workspace
      for key in ("out_status","out_diagnostics","out_acc","out_force","workspace_debug"):
        if key in w:entry[key]=array(w[key])
    entry["dense_status"]=array(sim._solver._status)
    entry["rhs_hi"]=array(sim._rhs);entry["rhs_low"]=array(sim._rhs_low)
    entry["mass_factor"]=array(sim._solver._factor)
    entry["smooth_hi"]=array(sim._solver._solution);entry["smooth_low"]=array(sim._solver._solution_low)
    records.append(entry)
    if any(status):break
  out=os.environ.get("MUJOCO_METAL_EVIDENCE_DIR")
  if out:
    d=Path(out);d.mkdir(parents=True,exist_ok=True)
    (d/"soft-stiff-first-failure.json").write_text(json.dumps({"xml":xml,"steps_requested":400,"records":records,"failed_calls":failed_calls},indent=2)+"\n")
  assert len(records)==400 and not any(records[-1]["status"]),records[-1]
  np.testing.assert_allclose(sim.state.qpos.detach().cpu().numpy()[0],cpu.qpos,atol=5e-3)
