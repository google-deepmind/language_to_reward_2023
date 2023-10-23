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

"""Python client for interfacing with an MJPC binary running the Barkour task."""

import dataclasses
import pathlib
from typing import Optional, Sequence
import mujoco
from mujoco_mpc import agent as agent_lib
import numpy as np
from language_to_reward_2023 import task_clients
from language_to_reward_2023.platforms.barkour import barkour_l2r_tasks


_MODEL_PATH = str(
    pathlib.Path(__file__).parent.parent.parent
        / "mjpc" / "barkour" / "world.xml"
)


@dataclasses.dataclass
class BarkourState:
  """Current state of the Barkour robot from the MJPC binary."""
  # Torso state (world space)
  torso_position: np.ndarray
  torso_linear_velocity: np.ndarray
  torso_orientation: np.ndarray
  torso_angular_velocity: np.ndarray

  # Foot space (local space)
  front_left_foot_position: np.ndarray
  front_right_foot_position: np.ndarray
  front_left_foot_velocity: np.ndarray
  front_right_foot_velocity: np.ndarray
  hind_left_foot_position: np.ndarray
  hind_right_foot_position: np.ndarray
  hind_left_foot_velocity: np.ndarray
  hind_right_foot_velocity: np.ndarray


class BarkourClient(task_clients.AgentApiTaskClient):
  """Client for controlling MJPC when it's running the Barkour task."""

  def __init__(
      self,
      ui: bool = False,
      agent: Optional[agent_lib.Agent] = None,
      test_params: Sequence[str] = (),  # Do not use, test only.
  ):
    agent = agent or task_clients.create_agent(
        task_id='Barkour',
        ui=ui,
        real_time_speed=0.4,
    )
    model = agent.model or mujoco.MjModel.from_xml_path(_MODEL_PATH)

    super().__init__(
        agent=agent,
        model=model,
        # normally, we run at 25% real time, and planning time is 50 to 80ms,
        # so 13ms planning delay should be fine, but for some reason performance
        # is not so good in this case, so we set a planning duration such that
        # planning happens every step.
        # TODO(nimrod): change this to 13ms.
        planning_duration=0.005,
    )

  def reset(self):
    super().reset()
    self._reset_reward()

  def get_state(self) -> BarkourState:
    self.update_state()
    return BarkourState(
        torso_position=np.array(self._data.sensor('torso_subtreecom').data),
        torso_orientation=np.array(self._data.sensor('torso_subtreequat').data),
        torso_linear_velocity=np.array(
            self._data.sensor('torso_subtreelinvel').data
        ),
        torso_angular_velocity=np.array(self._data.sensor('torso_angvel').data),
        front_left_foot_position=np.array(self._data.sensor('FLpos').data),
        hind_left_foot_position=np.array(self._data.sensor('RLpos').data),
        front_right_foot_position=np.array(self._data.sensor('FRpos').data),
        hind_right_foot_position=np.array(self._data.sensor('RRpos').data),
        front_left_foot_velocity=np.array(self._data.sensor('FLvel').data),
        hind_left_foot_velocity=np.array(self._data.sensor('RLvel').data),
        front_right_foot_velocity=np.array(self._data.sensor('FRvel').data),
        hind_right_foot_velocity=np.array(self._data.sensor('RRvel').data),
    )

  def _reset_reward(self):
    weights, params = barkour_l2r_tasks.defaults()
    self._agent.set_task_parameters(params)
    self._agent.set_cost_weights(weights)
