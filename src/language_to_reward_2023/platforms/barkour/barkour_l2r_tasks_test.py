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

from absl.testing import absltest
from absl.testing import parameterized
from mujoco_mpc import agent as mjpc_agent
import numpy as np

from language_to_reward_2023.platforms.barkour import barkour_l2r_task_client
from language_to_reward_2023.platforms.barkour import barkour_l2r_tasks

_AGENT = None


def setUpModule():
  global _AGENT
  _AGENT = barkour_l2r_task_client.BarkourClient().agent()


def tearDownModule():
  global _AGENT
  _AGENT.close()
  del _AGENT


class BarkourTasksTest(parameterized.TestCase):

  def test_defaults(self):
    weights, params = barkour_l2r_tasks.defaults()
    self.assertEqual(0, weights["FL_forward"])
    self.assertEqual(0, params["GoalPositionX"])
    self.assertAlmostEqual(0.292, params["TargetBodyHeight"])

  def test_set_torso_targets_location_xy(self):
    """Tests that setting torso targets results in the correct behavior."""
    weights, params = barkour_l2r_tasks.defaults()
    target = (1.0, 2.0)
    barkour_l2r_tasks.set_torso_targets(
        weights, params, target_torso_location_xy=target
    )
    state = self.run_agent(
        _AGENT,
        weights,
        params,
        until=lambda state: _distance(state.qpos[:2], target) < 0.2,
    )
    self.assertLessEqual(_distance(state.qpos[:2], target), 0.2)

  def run_agent(
      self,
      agent: mjpc_agent.Agent,
      weights,
      params,
      until=lambda x: False,
      max_time=4.0,
  ):
    agent.reset()
    agent.set_cost_weights(weights)
    agent.set_task_parameters(params)
    state = agent.get_state()
    while state.time < max_time and not until(state):
      agent.planner_step()
      for _ in range(10):
        agent.step()
      state = agent.get_state()
    return state


def _distance(a, b):
  return np.linalg.norm(np.array(a) - np.array(b))


if __name__ == "__main__":
  absltest.main()
