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

"""Abstract class for LLM Prompts."""
import dataclasses
from typing import Any, Optional, Sequence


@dataclasses.dataclass
class LLMPrompt:
  name: str = "TaskEvaluator"
  num_llms: int = 0
  prompts: Sequence[str] = dataclasses.field(default_factory=list)
  keep_message_history: Sequence[bool] = dataclasses.field(default_factory=list)
  response_processors: Sequence[Any] = dataclasses.field(default_factory=list)
  code_executor: Optional[Any] = None
