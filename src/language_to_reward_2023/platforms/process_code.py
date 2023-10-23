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

"""Utilities for processing LLM-generated code into executable code."""

import re


def _fix_code(code_str: str) -> str:
  """Fixes common mistakes in LLM-generated code."""
  if "np." in code_str and "import numpy as np" not in code_str:
    code_str = "import numpy as np\n" + code_str

  return code_str


def process_code_block(text: str) -> str:
  """Extracts code blocks from the input string.

  Arguments:
    text: A string which may contain markdown code blocks.

  Returns:
    Code concatenated from the markdown code blocks, with some code fixes
    applied. The code is untrusted and should only be executed in a sandbox.
  """
  code_block_regex = r"```(python)?\n?([\s\S]*?)```"
  matches = re.findall(code_block_regex, text)
  if matches:
    code = "\n".join([m[1] for m in matches])
  else:
    # If there are no code block markers, assume the whole text is one code
    # block.
    code = text
  return _fix_code(code)
