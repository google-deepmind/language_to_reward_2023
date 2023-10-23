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

"""Config file for evaluation and user interaction scripts."""

import dataclasses
from typing import Any

from language_to_reward_2023 import task_clients
from language_to_reward_2023.platforms.barkour import barkour_l2r_task_client
from language_to_reward_2023.platforms.barkour.prompts import prompt_coder_only as bk_prompt_coder_only
from language_to_reward_2023.platforms.barkour.prompts import prompt_low_level as bk_prompt_low_level
from language_to_reward_2023.platforms.barkour.prompts import prompt_thinker_coder as bk_prompt_thinker_coder


@dataclasses.dataclass(frozen=True)
class TaskConfig:
  client: type[task_clients.TaskClient]
  prompts: dict[str, type[Any]]


ALL_TASKS = {
    'barkour': TaskConfig(
        client=barkour_l2r_task_client.BarkourClient,
        prompts={
            'thinker_coder': bk_prompt_thinker_coder.PromptThinkerCoder,
            'coder_only': bk_prompt_coder_only.PromptCoder,
            'low_level': bk_prompt_low_level.PromptLowLevel,
        },
    ),
}
