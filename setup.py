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

"""Setup script for language_to_reward_2023."""

import os
import pathlib
import platform
import shutil
import subprocess

import setuptools
from setuptools.command import build_ext


Path = pathlib.Path


class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])


class BuildAgentServerBinaries(build_ext.build_ext):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def run(self):
    self._configure_and_build_agent_server()
    self.run_command("copy_agent_server_binary")
    self.run_command("copy_task_assets")

  def _configure_and_build_agent_server(self):
    """Check for CMake."""
    cmake_command = "cmake"
    build_cfg = "Debug"
    l2r_root = Path(__file__).parent
    l2r_build_dir = l2r_root / "build"
    cmake_configure_args = [
        f"-DCMAKE_BUILD_TYPE:STRING={build_cfg}",
        "-DBUILD_TESTING:BOOL=OFF",
    ]

    if platform.system() == "Darwin" and "ARCHFLAGS" in os.environ:
      osx_archs = []
      if "-arch x86_64" in os.environ["ARCHFLAGS"]:
        osx_archs.append("x86_64")
      if "-arch arm64" in os.environ["ARCHFLAGS"]:
        osx_archs.append("arm64")
      cmake_configure_args.append(
          f"-DCMAKE_OSX_ARCHITECTURES={';'.join(osx_archs)}"
      )

    # TODO(nimrod): We currently configure the builds into
    # `mujoco_mpc/build`. This should use `self.build_{temp,lib}` instead, to
    # isolate the Python builds from the C++ builds.
    print("Configuring CMake with the following arguments:")
    for arg in cmake_configure_args:
      print(f"  {arg}")

    subprocess.check_call(
        [
            cmake_command,
            *cmake_configure_args,
            f"-S{l2r_root.resolve()}",
            f"-B{l2r_build_dir.resolve()}",
        ],
        cwd=l2r_root,
    )

    print("Building `l2r_headless_server` and `l2r_ui_server` with CMake")
    subprocess.check_call(
        [
            cmake_command,
            "--build",
            str(l2r_build_dir.resolve()),
            "--target",
            "l2r_headless_server",
            "l2r_ui_server",
            f"-j{os.cpu_count()}",
            "--config",
            build_cfg,
        ],
        cwd=l2r_root,
    )


class CopyAgentServerBinaryCommand(setuptools.Command):
  """Command to copy `l2r_{headless,ui}_server` next to task_clients.py.

  Assumes that the C++ binaries were built and
  and located in the default `build/mjpc` folder.
  """

  description = "Copy server binaries into package."
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("copy_task_assets", ("build_lib", "build_lib"))

  def run(self):
    self._copy_binary("l2r_headless_server")
    self._copy_binary("l2r_ui_server")

  def _copy_binary(self, binary_name):
    source_path = Path(f"build/mjpc/{binary_name}")
    if not source_path.exists():
      raise ValueError(
          f"Cannot find `{binary_name}` binary from {source_path.absolute()}. "
          f"Please build the `{binary_name}` C++ gRPC service."
      )
    assert self.build_lib is not None
    build_lib_path = Path(self.build_lib).resolve()
    destination_path = Path(
        build_lib_path, "language_to_reward_2023", "mjpc", binary_name
    )

    self.announce(f"{source_path.resolve()=}")
    self.announce(f"{destination_path.resolve()=}")

    destination_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(source_path, destination_path)


class CopyTaskAssetsCommand(setuptools.Command):
  """Copies XML and mesh files next to the server binaries."""

  description = "Copy task assets over to python source."
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("build_py", ("build_lib", "build_lib"))

  def run(self):
    l2r_root = Path(__file__).parent
    l2r_build_dir = l2r_root / "build"
    mjpc_task_paths = [l2r_build_dir / "mjpc" / "barkour"]
    for task_path in mjpc_task_paths:
      relative_task_path = task_path.relative_to(l2r_build_dir)
      assert self.build_lib is not None
      build_lib_path = Path(self.build_lib).resolve()
      destination_dir_path = Path(build_lib_path, "language_to_reward_2023")
      self.announce(
          f"Copying assets {relative_task_path} from"
          f" {l2r_build_dir} over to {destination_dir_path}."
      )

      destination_path = destination_dir_path / relative_task_path
      shutil.copytree(task_path, destination_path, dirs_exist_ok=True)


setuptools.setup(
    ext_modules=[CMakeExtension("agent_server")],
    cmdclass={
        "build_ext": BuildAgentServerBinaries,
        "copy_agent_server_binary": CopyAgentServerBinaryCommand,
        "copy_task_assets": CopyTaskAssetsCommand,
    },
)
