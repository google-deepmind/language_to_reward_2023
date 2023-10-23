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

"""Execute untrusted code to get task parameters for the Barkour MJPC task."""

import abc
import json
from typing import Any, Callable, Sequence

from absl import logging
from mujoco_mpc import mjpc_parameters

from language_to_reward_2023 import safe_executor
from language_to_reward_2023.platforms.barkour import barkour_l2r_tasks


class BarkourSafeExecutor(metaclass=abc.ABCMeta):
  """Runs untrusted code to get task parameters for the Barkour MJPC task."""

  def __init__(self, executor: safe_executor.SafeExecutor):
    self._executor = executor

  def execute(self, code: str) -> mjpc_parameters.MjpcParameters:
    """Executes untrusted code to get MJPC task parameters.

    See _CODE_TEMPLATE below for available functions.

    Arguments:
      code: code which edits the weights and params dicts.

    Raises:
      ValueError: the code doesn't compile or failed to run.

    Returns:
      MJPC task parameters for the Barkour robot.
    """
    return self._execute_with_template(_CODE_TEMPLATE, code)

  def _execute_with_template(
      self, template: str, code: str
  ) -> mjpc_parameters.MjpcParameters:
    """Combines untrusted code with a template and executes it in a sandbox."""
    combined_code = template.replace('INSERT_CODE_HERE', code)
    output = self._executor.safe_execute(combined_code)
    return _parse_output(output)


def _parse_output(output: str) -> mjpc_parameters.MjpcParameters:
  """Parses the output from executing param generation code."""
  try:
    overwriting_weights, overwriting_params = json.loads(output)
  except ValueError as e:
    raise ValueError(f'Invalid JSON output: {output}') from e

  weights, params = barkour_l2r_tasks.defaults()
  _overwrite_entries(weights, overwriting_weights, 'weight')
  _overwrite_entries(params, overwriting_params, 'param')
  return mjpc_parameters.MjpcParameters(
      cost_weights=weights, task_parameters=params
  )


def _overwrite_entries(base, overwrites, name_for_log) -> None:
  for key, value in overwrites.items():
    if key not in base:
      logging.warning('Invalid %s: %s', name_for_log, key)
      continue
    try:
      base[key] = float(value)
    except ValueError:
      logging.warning('Invalid float value for %s: %s', key, value)


def _generate_function_definitions(
    supported_functions: Sequence[Callable[..., Any]]
) -> str:
  """Generates function definitions for supported functions."""
  definitions = []
  for function in supported_functions:
    definitions.append(f"""
def {function.__name__}(*args, **kwargs):
  global weights, params
  barkour_l2r_tasks.{function.__name__}(weights, params, *args, **kwargs)


""")
  for dummy_function_name in _DUMMY_FUNCTIONS:
    definitions.append(f"""
def {dummy_function_name}(*args, **kwargs):
  pass

""")

  return ''.join(definitions)


_SUPPORTED_FUNCTIONS = (
    barkour_l2r_tasks.set_torso_targets,
    barkour_l2r_tasks.set_foot_pos_parameters,
    barkour_l2r_tasks.set_foot_stepping_parameters,
    barkour_l2r_tasks.set_target_joint_angles,
    barkour_l2r_tasks.head_towards,
    barkour_l2r_tasks.walk,
)

_DUMMY_FUNCTIONS = (
    'reset_reward',
    'execute_plan',
)


_CODE_TEMPLATE = r"""
from language_to_reward_2023.platforms.barkour import barkour_l2r_tasks

weights, params = barkour_l2r_tasks.defaults()

FUNCTION_DEFINITIONS

INSERT_CODE_HERE

import json
print(json.JSONEncoder().encode((weights, params)))
""".replace(
    'FUNCTION_DEFINITIONS', _generate_function_definitions(_SUPPORTED_FUNCTIONS)
)
