# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Classes that manage resources related to MJPC tasks."""

import abc
import pathlib
from typing import Any, Callable, Mapping, Optional, Sequence

import mujoco
from mujoco_mpc import agent as agent_lib
import numpy as np

from mujoco_mpc.proto import agent_pb2


DEFAULT_UI_SERVER_PATH = str(
    pathlib.Path(__file__).parent / "mjpc" / "l2r_ui_server"
)
DEFAULT_HEADLESS_SERVER_PATH = str(
    pathlib.Path(__file__).parent / "mjpc" / "l2r_headless_server"
)


class TaskClient(abc.ABC):
  """Base class for task clients."""

  @abc.abstractmethod
  def reset(self):
    """Resets the task to a starting position."""

  @abc.abstractmethod
  def model(self) -> mujoco.MjModel:
    """Returns an MjModel for the task."""

  def close(self):
    """Releases any resources associated with the task."""

  def methods(self) -> dict[str, Callable[..., Any]]:
    """Returns the set of methods supported by the client."""
    return {}


def create_agent(
    task_id: str,
    ui: bool,
    real_time_speed: float = 1,
    server_binary_path: Optional[str] = None,
    subprocess_kwargs: Optional[Mapping[str, Any]] = None,
    extra_flags: Optional[Sequence[str]] = None,
) -> agent_lib.Agent:
  """Helper function to create an agent_lib.Agent instance."""
  if server_binary_path is None:
    if ui:
      server_binary_path = DEFAULT_UI_SERVER_PATH
    else:
      server_binary_path = DEFAULT_HEADLESS_SERVER_PATH

  if ui:
    flags = [
        '--planner_enabled',
    ]
  else:
    flags = []

  return agent_lib.Agent(
      task_id,
      server_binary_path=server_binary_path,
      real_time_speed=real_time_speed,
      extra_flags=flags + (extra_flags or []),
      subprocess_kwargs=subprocess_kwargs,
  )


class AgentApiTaskClient(TaskClient):
  """Base class for task clients which starts the Agent Service.

  By default, this will be a headless client, but if ui==True, an interactive
  user interface will be run while executing the task. The UI version will
  behave differently from the headless version, because planning is async.
  """

  def __init__(
      self,
      agent: agent_lib.Agent,
      model: mujoco.MjModel,
      planning_duration: float = 0.04,
      real_time_speed: float = 1.0,
  ):
    self._agent = agent
    self._model = model
    self._data = mujoco.MjData(self._model)

    self._last_planning_time = None
    self._planning_duration = planning_duration
    self._real_time_speed = real_time_speed
    self._recorded = None

  def agent(self) -> agent_lib.Agent:
    return self._agent

  def model(self) -> mujoco.MjModel:
    return self._model

  def reset(self):
    self._agent.reset()
    self._last_planning_time = None

  def close(self):
    """Releases any resources associated with the task."""
    self._agent.close()

  def start_recording(self):
    """Start recording simulation states, for later rendering."""
    self._recorded = []

  def end_recording(self) -> list[agent_pb2.State] | None:
    """Returns a list of states recorded since the call to start_recording."""
    recorded = self._recorded
    self._recorded = None
    return recorded

  def update_state(self):
    """Updates self._data with latest state from the agent binary."""
    state = self._agent.get_state()
    self._data.time = state.time
    self._data.qpos = state.qpos
    self._data.qvel = state.qvel
    self._data.act = state.act
    self._data.mocap_pos = np.array(state.mocap_pos).reshape(
        self._data.mocap_pos.shape
    )
    self._data.mocap_quat = np.array(state.mocap_quat).reshape(
        self._data.mocap_quat.shape
    )
    self._data.userdata = np.array(state.userdata).reshape(
        self._data.userdata.shape
    )
    mujoco.mj_forward(self._model, self._data)
    if self._recorded is not None:
      self._recorded.append(state)

  def execute_plan(self, duration=10):
    """Runs MJPC for `duration` seconds while executing planner actions."""
    state = self._agent.get_state()
    start = state.time
    while state.time - start < duration:
      if self._last_planning_time is None or (
          state.time - self._last_planning_time
          >= self._planning_duration * self._real_time_speed
      ):
        self._last_planning_time = state.time
        # repeat planning for 5 times
        for _ in range(5):
          self._agent.planner_step()
      self.update_state()
      self._agent.set_state()  # needed to call transition()
      self._agent.step()
      state = self._agent.get_state()
