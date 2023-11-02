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

"""Base class for executing untrusted code."""

from typing import Protocol


class SafeExecutor(Protocol):
  """Base class for executors that run untrusted code and produce their output."""

  def safe_execute(self, code: str) -> str:
    """Executes the given Python code and returns the standard output from it.

    Arguments:
      code: code which edits the weights and params dicts.

    Raises:
      ValueError: the code doesn't compile or failed to run.

    Returns:
      The standard output from the executed code.
    """
