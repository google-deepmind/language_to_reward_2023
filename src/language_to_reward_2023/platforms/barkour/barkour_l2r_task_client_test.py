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

"""Tests for Barkour task client."""

from absl.testing import absltest

from language_to_reward_2023.platforms.barkour import barkour_l2r_task_client


class BarkourTaskClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.client = barkour_l2r_task_client.BarkourClient()

  def tearDown(self):
    self.client.close()
    super().tearDown()

  def test_creates_agent(self):
    self.assertIsNotNone(self.client.agent())


if __name__ == '__main__':
  absltest.main()
