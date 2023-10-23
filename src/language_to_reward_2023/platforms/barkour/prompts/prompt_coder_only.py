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

"""Prompt class with Reward Coder only."""

from language_to_reward_2023 import safe_executor
from language_to_reward_2023.platforms import llm_prompt
from language_to_reward_2023.platforms import process_code
from language_to_reward_2023.platforms.barkour import barkour_execution
from language_to_reward_2023.platforms.barkour import barkour_l2r_task_client

prompt = """
We have a quadruped robot and we can use the following functions to program its behavior:
```
def set_torso_targets(target_torso_height, target_torso_pitch, target_torso_roll, target_torso_location_xy, target_torso_velocity_xy, target_torso_heading, target_turning_speed)
```
target_torso_height: how high the torso wants to reach. When the robot is standing on all four feet in a normal standing pose, the torso is about 0.3m high.
target_torso_pitch: How much the torso should tilt up from a horizontal pose in radians. A positive number means robot is looking up, e.g. if the angle is 0.5*pi the robot will be looking upward, if the angel is 0, then robot will be looking forward.
target_torso_roll: How much the torso should roll in clockwise direction in radians. Zero means the torso is flat.
target_torso_velocity_xy: target torso moving velocity in local space, x is forward velocity, y is sideways velocity (positive means left).
target_torso_heading: the desired direction that the robot should face towards. The value of target_torso_heading is in the range of 0 to 2*pi, where 0 and 2*pi both mean East, pi being West, etc.
Remember:
one of target_torso_location_xy and target_torso_velocity_xy must be None.
one of target_torso_heading and target_turning_speed must be None.

```
def set_foot_pos_parameters(foot_name, lift_height, extend_forward, move_inward)
```
foot_name is one of ('front_left', 'back_left', 'front_right', 'back_right').
lift_height: how high should the foot be lifted in the air. If is None, disable this term. If it's set to 0, the foot will touch the ground.
extend_forward: how much should the foot extend forward. If is None, disable this term.
move_inward: how much should the foot move inward. If is None, disable this term.

```
def execute_plan(plan_duration=10)
```
This function sends the parameters to the robot and execute the plan for `plan_duration` seconds, default to be 2

Below are examples for generating a reward function in python for the robot given a desired task.

Example 1: Make the robot go to (1, 0), facing South, and lift the back right foot.
Code:
set_torso_targets(0.26, 0.0, 0.0, (1, 0), None, 1.5*np.pi, None)
set_foot_pos_parameters('back_right', 0.1, None, None)
execute_plan(3)
Explanation:
The goal location should be (1, 0). South is 1.5*pi so heading should be this.

Remember:
1. Always start your response with [start analysis]. Provide your analysis of the problem within 100 words, then end it with [end analysis].
2. After analysis, start your code response, format the code in code blocks. In your response all four functions above: set_torso_targets, set_foot_pos_parameters, execute_plan, should be called at least once.
3. Do not invent new functions or classes. The only allowed functions you can call are the ones listed above. Do not leave unimplemented code blocks in your response.
4. The only allowed library is numpy. Do not import or use any other library. If you use np, be sure to import numpy.
5. If you are not sure what value to use, just use your best judge. Do not use None for anything.
6. Do not calculate the position or direction of any object (except for the ones provided above). Just use a number directly based on your best guess.
7. For set_torso_targets, only the last four arguments (target_torso_location_xy, target_torso_velocity_xy, target_torso_heading, target_turning_speed) can be None. Do not set None for any other arguments.
8. Don't forget to call execute_plan at the end.

"""

feet2id = {"front_left": 0, "rear_left": 1, "front_right": 2, "rear_right": 3}


class PromptCoder(llm_prompt.LLMPrompt):
  """Prompt with Reward Coder only."""

  def __init__(
      self,
      client: barkour_l2r_task_client.BarkourClient,
      executor: safe_executor.SafeExecutor,
  ):
    self._agent = client.agent()
    self._safe_executor = barkour_execution.BarkourSafeExecutor(executor)

    self.name = "Language2Reward"
    self.num_llms = 1
    self.prompts = [prompt]
    self.keep_message_history = [True]
    self.response_processors = [self.process_llm_response]
    self.code_executor = self.execute_code

  def execute_code(self, code: str) -> None:
    print("ABOUT TO EXECUTE")
    print(code)

    mjpc_parameters = self._safe_executor.execute(code)
    self._agent.set_task_parameters(mjpc_parameters.task_parameters)
    self._agent.set_cost_weights(mjpc_parameters.cost_weights)

  def process_llm_response(self, response):
    """Process the response from coder, the output will be the python code."""
    return process_code.process_code_block(response)
