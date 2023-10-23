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

from absl.testing import absltest
from language_to_reward_2023.platforms import process_code


class ProcessCodeTest(absltest.TestCase):

  def test_extract_code_block(self):
    block = process_code.process_code_block("""
```python
banana
```
""")
    self.assertEqual(block.strip(), "banana")

  def test_extract_multiple_code_blocks(self):
    block = process_code.process_code_block("""
```python
banana
```

```
apple
```
""")
    self.assertEqual(block, "banana\n\napple\n")

  def test_add_numpy(self):
    block = process_code.process_code_block("""
```python
np.zeros(3)
```
""")
    self.assertEqual(block.strip(), "import numpy as np\nnp.zeros(3)")

  def test_no_block_markers(self):
    block = process_code.process_code_block("""np.zeros(3)
""")
    self.assertEqual(block.strip(), "import numpy as np\nnp.zeros(3)")


if __name__ == "__main__":
  absltest.main()
