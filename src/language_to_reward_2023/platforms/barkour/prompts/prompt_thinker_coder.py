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

"""Prompt class with both Motion Descriptor and Reward Coder."""
import re

from language_to_reward_2023 import safe_executor
from language_to_reward_2023.platforms import llm_prompt
from language_to_reward_2023.platforms import process_code
from language_to_reward_2023.platforms.barkour import barkour_execution
from language_to_reward_2023.platforms.barkour import barkour_l2r_task_client

prompt_thinker = """
Describe the motion of a dog robot using the following form:

[start of description]
* This {CHOICE: [is, is not]} a new task.
* The torso of the robot should roll by [NUM: 0.0] degrees towards right, the torso should pitch upward at [NUM: 0.0] degrees.
* The height of the robot's CoM or torso center should be at [NUM: 0.3] meters.
* The robot should {CHOICE: [face certain direction, turn at certain speed]}. If facing certain direction, it should be facing {CHOICE: [east, south, north, west]}. If turning, it should turn at [NUM: 0.0] degrees/s.
* The robot should {CHOICE: [go to a certain location, move at certain speed]}. If going to certain location, it should go to (x=[NUM: 0.0], y=[NUM: 0.0]). If moving at certain speed, it should move forward at [NUM: 0.0]m/s and sideways at [NUM: 0.0]m/s (positive means left).
* [optional] front_left foot lifted to [NUM: 0.0] meters high.
* [optional] back_left foot lifted to [NUM: 0.0] meters high.
* [optional] front_right foot lifted to [NUM: 0.0] meters high.
* [optional] back_right foot lifted to [NUM: 0.0] meters high.
* [optional] front_left foot extend forward by [NUM: 0.0] meters.
* [optional] back_left foot extend forward by [NUM: 0.0] meters.
* [optional] front_right foot extend forward by [NUM: 0.0] meters.
* [optional] back_right foot extend forward by [NUM: 0.0] meters.
* [optional] front_left foot shifts inward laterally by [NUM: 0.0] meters.
* [optional] back_left foot shifts inward laterally by [NUM: 0.0] meters.
* [optional] front_right foot shifts inward laterally by [NUM: 0.0] meters.
* [optional] back_right foot shifts inward laterally by [NUM: 0.0] meters.
* [optional] front_left foot steps on the ground at a frequency of [NUM: 0.0] Hz, during the stepping motion, the foot will move [NUM: 0.0] meters up and down, and [NUM: 0.0] meters forward and back, drawing a circle as if it's walking {CHOICE: forward, back}, spending [NUM: 0.0] portion of the time in the air vs gait cycle.
* [optional] back_left foot steps on the ground at a frequency of [NUM: 0.0] Hz, during the stepping motion, the foot will move [NUM: 0.0] meters up and down, and [NUM: 0.0] meters forward and back, drawing a circle as if it's walking {CHOICE: forward, back}, spending [NUM: 0.0] portion of the time in the air vs gait cycle.
* [optional] front_right foot steps on the ground at a frequency of [NUM: 0.0] Hz, during the stepping motion, the foot will move [NUM: 0.0] meters up and down, and [NUM: 0.0] meters forward and back, drawing a circle as if it's walking {CHOICE: forward, back}, spending [NUM: 0.0] portion of the time in the air vs gait cycle.
* [optional] back_right foot steps on the ground at a frequency of [NUM: 0.0] Hz, during the stepping motion, the foot will move [NUM: 0.0] meters up and down, and [NUM: 0.0] meters forward and back, drawing a circle as if it's walking {CHOICE: forward, back}, spending [NUM: 0.0] portion of the time in the air vs gait cycle.
* [optional] The phase offsets for the four legs should be front_left: [NUM: 0.0], back_left: [NUM: 0.0], front_right: [NUM: 0.0], back_right: [NUM: 0.0].

[end of description]

Rules:
1. If you see phrases like [NUM: default_value], replace the entire phrase with a numerical value. If you see [PNUM: default_value], replace it with a positive, non-zero numerical value.
2. If you see phrases like {CHOICE: [choice1, choice2, ...]}, it means you should replace the entire phrase with one of the choices listed. Be sure to replace all of them. If you are not sure about the value, just use your best judgement.
3. Phase offset is between [0, 1]. So if two legs' phase offset differs by 0 or 1 they are moving in synchronous. If they have phase offset difference of 0.5, they are moving opposite in the gait cycle.
4. The portion of air vs the gait cycle is between [0, 1]. So if it's 0, it means the foot will always stay on the ground, and if it's 1 it means the foot will always be in the air.
5. I will tell you a behavior/skill/task that I want the quadruped to perform and you will provide the full description of the quadruped motion, even if you may only need to change a few lines. Always start the description with [start of description] and end it with [end of description].
6. We can assume that the robot has a good low-level controller that maintains balance and stability as long as it's in a reasonable pose.
7. You can assume that the robot is capable of doing anything, even for the most challenging task.
8. The robot is about 0.3m high in CoM or torso center when it's standing on all four feet with horizontal body. It's about 0.65m high when it stand upright on two feet with vertical body. When the robot's torso/body is flat and parallel to the ground, the pitch and roll angles are both 0.
9. Holding a foot 0.0m in the air is the same as saying it should maintain contact with the ground.
10. Do not add additional descriptions not shown above. Only use the bullet points given in the template.
11. If a bullet point is marked [optional], do NOT add it unless it's absolutely needed.
12. Use as few bullet points as possible. Be concise.

"""

prompt_coder = """
We have a description of a robot's motion and we want you to turn that into the corresponding program with following functions:
```
def set_torso_targets(target_torso_height, target_torso_pitch, target_torso_roll, target_torso_location_xy, target_torso_velocity_xy, target_torso_heading, target_turning_speed)
```
target_torso_height: how high the torso wants to reach. When the robot is standing on all four feet in a normal standing pose, the torso is about 0.3m high.
target_torso_pitch: How much the torso should tilt up from a horizontal pose in radians. A positive number means robot is looking up, e.g. if the angle is 0.5*pi the robot will be looking upward, if the angel is 0, then robot will be looking forward.
target_torso_velocity_xy: target torso moving velocity in local space, x is forward velocity, y is sideways velocity (positive means left).
target_torso_heading: the desired direction that the robot should face towards. The value of target_torso_heading is in the range of 0 to 2*pi, where 0 and 2*pi both mean East, pi being West, etc.
target_turning_speed: the desired turning speed of the torso in radians per second.
Remember:
one of target_torso_location_xy and target_torso_velocity_xy must be None.
one of target_torso_heading and target_turning_speed must be None.
No other inputs can be None.

```
def set_foot_pos_parameters(foot_name, lift_height, extend_forward, move_inward)
```
foot_name is one of ('front_left', 'back_left', 'front_right', 'back_right').
lift_height: how high should the foot be lifted in the air. If is None, disable this term. If it's set to 0, the foot will touch the ground.
extend_forward: how much should the foot extend forward. If is None, disable this term.
move_inward: how much should the foot move inward. If is None, disable this term.

```
def set_foot_stepping_parameters(foot_name, stepping_frequency, air_ratio, phase_offset, swing_up_down, swing_forward_back, should_activate)
```
foot_name is one of ('front_left', 'rear_left', 'front_right', 'rear_right').
air_ratio (value from 0 to 1) describes how much time the foot spends in the air versus the whole gait cycle. If it's 0 the foot will always stay on ground, and if it's 1 it'll always stay in the air.
phase_offset (value from 0 to 1) describes how the timing of the stepping motion differs between
different feet. For example, if the phase_offset between two legs differs by 0.5, it means
one leg will start the stepping motion in the middle of the stepping motion cycle of the other leg.
swing_up_down is how much the foot swings vertical during the motion cycle.
swing_forward_back is how much the foot swings horizontally during the motion cycle.
If swing_forward_back is positive, the foot would look like it's going forward, if it's negative, the foot will look like it's going backward.
If should_activate is False, the leg will not follow the stepping motion.

```
def execute_plan(plan_duration=2)
```
This function sends the parameters to the robot and execute the plan for `plan_duration` seconds, default to be 2

Example answer code:
```
import numpy as np  # import numpy because we are using it below

reset_reward() # This is a new task so reset reward; otherwise we don't need it
set_torso_targets(0.1, np.deg2rad(5), np.deg2rad(15), (2, 3), None, None, np.deg2rad(10))

set_foot_pos_parameters('front_left', 0.1, 0.1, None)
set_foot_pos_parameters('back_left', None, None, 0.15)
set_foot_pos_parameters('front_right', None, None, None)
set_foot_pos_parameters('back_right', 0.0, 0.0, None)
set_foot_stepping_parameters('front_right', 2.0, 0.5, 0.2, 0.1, -0.05, True)
set_foot_stepping_parameters('back_left', 3.0, 0.7, 0.1, 0.1, 0.05, True)
set_foot_stepping_parameters('front_left', 0.0, 0.0, 0.0, 0.0, 0.0, False)
set_foot_stepping_parameters('back_right', 0.0, 0.0, 0.0, 0.0, 0.0, False)

execute_plan(4)
```

Remember:
1. Always format the code in code blocks. In your response all four functions above: set_torso_targets, set_foot_pos_parameters, execute_plan, should be called at least once.
2. Do not invent new functions or classes. The only allowed functions you can call are the ones listed above. Do not leave unimplemented code blocks in your response.
3. The only allowed library is numpy. Do not import or use any other library. If you use np, be sure to import numpy.
4. If you are not sure what value to use, just use your best judge. Do not use None for anything.
5. Do not calculate the position or direction of any object (except for the ones provided above). Just use a number directly based on your best guess.
6. For set_torso_targets, only the last four arguments (target_torso_location_xy, target_torso_velocity_xy, target_torso_heading, target_turning_speed) can be None. Do not set None for any other arguments.
7. Don't forget to call execute_plan at the end.

"""


class PromptThinkerCoder(llm_prompt.LLMPrompt):
  """Prompt with both Motion Descriptor and Reward Coder."""

  def __init__(
      self,
      client: barkour_l2r_task_client.BarkourClient,
      executor: safe_executor.SafeExecutor,
  ):
    self._agent = client.agent()
    self._safe_executor = barkour_execution.BarkourSafeExecutor(executor)

    self.name = "Language2StructuredLang2Reward"

    self.num_llms = 2
    self.prompts = [prompt_thinker, prompt_coder]

    # The coder doesn't need to keep the history as it only serves a purpose for
    # translating to code
    self.keep_message_history = [True, False]
    self.response_processors = [
        self.process_thinker_response,
        self.process_coder_response,
    ]
    self.code_executor = self.execute_code

  # process the response from thinker, the output will be used as input to coder
  def process_thinker_response(self, response: str) -> str:
    try:
      motion_description = (
          re.split(
              "end of description",
              re.split("start of description", response, flags=re.IGNORECASE)[
                  1
              ],
              flags=re.IGNORECASE,
          )[0]
          .strip("[")
          .strip("]")
          .strip()
          .strip("```")
      )
      return motion_description
    except Exception as _:  # pylint: disable=broad-exception-caught
      return response

  def process_coder_response(self, response):
    """Process the response from coder, the output will be the python code."""
    return process_code.process_code_block(response)

  def execute_code(self, code: str) -> None:
    print("ABOUT TO EXECUTE\n", code)
    mjpc_parameters = self._safe_executor.execute(code)
    self._agent.set_task_parameters(mjpc_parameters.task_parameters)
    self._agent.set_cost_weights(mjpc_parameters.cost_weights)
