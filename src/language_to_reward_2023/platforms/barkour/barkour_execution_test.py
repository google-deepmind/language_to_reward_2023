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

import inspect

from absl.testing import absltest
from absl.testing import parameterized

from language_to_reward_2023 import confirmation_safe_executor
from language_to_reward_2023 import sandbox_safe_executor
from language_to_reward_2023.platforms.barkour import barkour_execution
from language_to_reward_2023.platforms.barkour import barkour_l2r_tasks


def _make_confirmation_executor():
  return barkour_execution.BarkourSafeExecutor(
      confirmation_safe_executor.ConfirmationSafeExecutor(
          skip_confirmation=True
      )
  )


def _make_sandbox_executor():
  return barkour_execution.BarkourSafeExecutor(
      sandbox_safe_executor.SandboxSafeExecutor()
  )


_EXECUTER_BUILDERS = [
    (_make_confirmation_executor,),
    (_make_sandbox_executor,),
]


class BarkourExecutionTest(parameterized.TestCase):

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_empty_code_returns_default_weights(self, executor_builder):
    executor = executor_builder()
    params = executor.execute('')
    default_cost_weights, default_task_params = barkour_l2r_tasks.defaults()
    self.assertEqual(default_cost_weights, params.cost_weights)
    self.assertEqual(default_task_params, params.task_parameters)

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_invalid_code_fails(self, executor_builder):
    executor = executor_builder()
    code = '('
    self.assertRaises(ValueError, lambda: executor.execute(code))

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_throwing_code_fails(self, executor_builder):
    executor = executor_builder()
    code = 'raise Exception()'
    self.assertRaises(ValueError, lambda: executor.execute(code))

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_printing_code_fails(self, executor_builder):
    executor = executor_builder()
    code = 'print("hello")'
    self.assertRaises(ValueError, lambda: executor.execute(code))

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_can_import_numpy(self, executor_builder):
    executor = executor_builder()
    params = executor.execute('import numpy as np')
    default_cost_weights, default_task_params = barkour_l2r_tasks.defaults()
    self.assertEqual(default_cost_weights, params.cost_weights)
    self.assertEqual(default_task_params, params.task_parameters)

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_invalid_parameters_ignored(self, executor_builder):
    executor = executor_builder()
    _, default_task_params = barkour_l2r_tasks.defaults()
    params = executor.execute('params["banana"] = 0.0')
    self.assertEqual(default_task_params, params.task_parameters)

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_invalid_float_ignored(self, executor_builder):
    executor = executor_builder()
    _, default_task_params = barkour_l2r_tasks.defaults()
    params = executor.execute('params["TargetBodyHeight"] = "asdf"')
    self.assertEqual(default_task_params, params.task_parameters)

  @parameterized.parameters(_EXECUTER_BUILDERS)
  def test_set_torso_targets(self, executor_builder):
    executor = executor_builder()
    code = """
set_torso_targets(target_torso_height=0.5)
"""
    params = executor.execute(code)
    self.assertEqual(0.5, params.task_parameters['TargetBodyHeight'])

  def test_all_supported_functions_are_valid(self):
    for f in barkour_execution._SUPPORTED_FUNCTIONS:
      args_names = list(inspect.signature(f).parameters.keys())
      self.assertEqual(
          ['cost_weights', 'task_parameters'],
          args_names[:2],
          f"First 2 parameters of {f.__name__} should be ['cost_weights',"
          " 'task_parameters']",
      )


if __name__ == '__main__':
  absltest.main()
