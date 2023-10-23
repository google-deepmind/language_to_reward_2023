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

"""Prompt for low level baseline."""

from language_to_reward_2023 import safe_executor
from language_to_reward_2023.platforms import llm_prompt
from language_to_reward_2023.platforms import process_code
from language_to_reward_2023.platforms.barkour import barkour_execution
from language_to_reward_2023.platforms.barkour import barkour_l2r_task_client

prompt = """
We have a quadruped robot. It has 12 joints in total, three for each leg.
We can use the following functions to control its movements:

```
def set_target_joint_angles(leg_name, target_joint_angles)
```
leg_name is one of ('front_left', 'back_left', 'front_right', 'back_right').
target_joint_angles: a 3D vector that describes the target angle for the abduction/adduction, hip, and knee joint of the each leg.

```
def walk(forward_speed, sideways_speed, turning_speed)
```
forward_speed: how fast the robot should walk forward
sideways_speed: how fast the robot should walk sideways
turning_speed: how fast the robot should be turning (positive means turning right)

```
def head_towards(heading_direction)
```
heading_direction: target heading for the robot to reach, in the range of 0 to 2*pi, where 0 means East, 0.5pi means North, pi means West, and 1.5pi means South.

```
def execute_plan(plan_duration=10)
```
This function sends the parameters to the robot and execute the plan for `plan_duration` seconds, default to be 2


Details about joint angles of each leg:
abduction/adduction joint controls the upper leg to swinging inward/outward.
When it's positive, legs will swing outward (swing to the right for right legs and left for left legs).
When it's negative, legs will swing inward.

hip joint controls the upper leg to rotate around the shoulder.
When it's zero, the upper leg is parallel to the torso (hip is same height as shoulder), pointing backward.
When it's positive, the upper leg rotates downward so the knee is below the shoulder. When it's 0.5pi, it's perpendicular to the torso, pointing downward.
When it's negative, the upper leg rotates upward so the knee is higher than the shoulder.

knee joint controls the lower leg to rotate around the knee.
When it's zero, the lower leg is folded closer to the upper leg.
knee joint angle can only be positive. When it's 0.5pi, the lower leg is perpendicular to the upper leg. When it's pi, the lower leg is fully streching out and parallel to the upper leg.

Here are a few examples for setting the joint angles to make the robot reach a few key poses:
standing on all four feet:
```
set_target_joint_angles("front_left", [0, 1, 1.5])
set_target_joint_angles("back_left", [0, 0.75, 1.5])
set_target_joint_angles("front_right", [0, 1, 1.5])
set_target_joint_angles("back_right", [0, 0.75, 1.5])
execute_plan()
```

sit down on the floor:
```
set_target_joint_angles("front_left", [0, 0, 0])
set_target_joint_angles("back_left", [0, 0, 0])
set_target_joint_angles("front_right", [0, 0, 0])
set_target_joint_angles("back_right", [0, 0, 0])
execute_plan()
```

lift front left foot:
```
set_target_joint_angles("front_left", [0, 0.45, 0.35])
set_target_joint_angles("back_left", [0, 1, 1.5])
set_target_joint_angles("front_right", [0, 1.4, 1.5])
set_target_joint_angles("back_right", [0, 1, 1.5])
execute_plan()
```

lift back left foot:
```
set_target_joint_angles("front_left", [0, 0.5, 1.5])
set_target_joint_angles("back_left", [0, 0.45, 0.35])
set_target_joint_angles("front_right", [0, 0.5, 1.5])
set_target_joint_angles("back_right", [0, 0.5, 1.5])
execute_plan()
```


Remember:
1. Always start your response with [start analysis]. Provide your analysis of the problem within 100 words, then end it with [end analysis].
2. After analysis, start your code response, format the code in code blocks.
3. Do not invent new functions or classes. The only allowed functions you can call are the ones listed above. Do not leave unimplemented code blocks in your response.
4. The only allowed library is numpy. Do not import or use any other library. If you use np, be sure to import numpy.
5. If you are not sure what value to use, just use your best judge. Do not use None for anything.
6. Do not calculate the position or direction of any object (except for the ones provided above). Just use a number directly based on your best guess.
7. Write the code as concisely as possible and try not to define additional variables.
8. If you define a new function for the skill, be sure to call it somewhere.
9. Be sure to call execute_plan at the end.

"""

feet2id = {"front_left": 0, "rear_left": 1, "front_right": 2, "rear_right": 3}


class PromptLowLevel(llm_prompt.LLMPrompt):
  """Prompt for low level baseline."""

  def __init__(
      self,
      client: barkour_l2r_task_client.BarkourClient,
      executor: safe_executor.SafeExecutor,
  ):
    self._agent = client.agent()
    self._safe_executor = barkour_execution.BarkourSafeExecutor(executor)

    self.name = "Language2Reward low level baseline"
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
    """Process the response from LLM, the output will be the python code."""
    return process_code.process_code_block(response)
