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

"""A class which manages a chat session using a prompt."""

import time
from typing import Any

import openai
import termcolor

from language_to_reward_2023.platforms import llm_prompt


def _open_ai_call_with_retry(
    model: str, messages: list[Any]
) -> openai.ChatCompletion:
  """Call OpenAI API with retry."""
  reset_trigger_phrases = ["CHOICE", "NUM"]
  success = False
  completion = None
  while not success:
    try:
      completion = openai.ChatCompletion.create(
          model=model,
          messages=messages,
          temperature=0.3,
      )
      success = True
      for reset_trigger_phrase in reset_trigger_phrases:
        if reset_trigger_phrase in completion.choices[0].message.content:
          success = False
    except Exception as e:  # pylint: disable=broad-exception-caught
      print("OpenAI API call issue, re-trying..." + str(e) + "\n")
      time.sleep(5)
  return completion


class Conversation:
  """Manages the state of a conversation with a language model."""

  def __init__(
      self,
      prompt_model: llm_prompt.LLMPrompt,
      model: str,
      print_responses: bool = True,
  ):
    self._prompt_model = prompt_model
    self._model = model
    self._print_responses = print_responses
    number_of_llms = prompt_model.num_llms
    self._message_queues = [[] for _ in range(number_of_llms)]

    # Add general prompt to the message queue.
    for llm_id in range(number_of_llms):
      message = [{"role": "user", "content": prompt_model.prompts[llm_id]}]
      self._message_queues[llm_id].append(message[0])

  def send_command(self, user_command: str) -> str:
    """Sends a user command to the LLMs, returns final processed response."""
    if user_command == "reset":
      self.reset()
      print("Resetting the conversation history.")
      return "reset"
    upstream_message = user_command + " Make sure to ignore irrelevant options."
    for llm_id in range(self._prompt_model.num_llms):
      completion = _open_ai_call_with_retry(
          self._model,
          self._message_queues[llm_id]
          + [{"role": "user", "content": upstream_message}],
      )
      if self._prompt_model.keep_message_history[llm_id]:
        self._message_queues[llm_id].append(
            {"role": "user", "content": upstream_message}
        )
        self._message_queues[llm_id].append(completion.choices[0].message)
      print(f"LLM{llm_id} queried")
      response = completion.choices[0].message.content
      if self._print_responses:
        print(termcolor.colored(response + "\n", "cyan", attrs=["bold"]))
      try:
        upstream_message = self._prompt_model.response_processors[llm_id](
            response
        )
      except Exception:
        if self._prompt_model.keep_message_history[llm_id]:
          self._message_queues[llm_id].pop(-1)
          self._message_queues[llm_id].pop(-1)
        raise
    return upstream_message

  def reset(self) -> None:
    for i in range(self._prompt_model.num_llms):
      while len(self._message_queues[i]) > 1:
        self._message_queues[i].pop(-1)
