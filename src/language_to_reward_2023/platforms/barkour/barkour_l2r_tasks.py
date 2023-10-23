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

"""Utilities for generating task parameters for the Barkour MJPC task."""

from typing import Literal

from language_to_reward_2023.platforms.barkour import default_task_parameters

FootName = Literal[
    'front_left',
    'rear_left',
    'front_right',
    'rear_right',
    'back_left',
    'back_right',
]


def defaults() -> tuple[dict[str, float], dict[str, float]]:
  """Default cost weights and task parameters for the Barkour MJPC task."""
  weights, params = default_task_parameters.default_parameters_and_weights()
  return weights, params


def set_torso_targets(
    cost_weights: dict[str, float],
    task_parameters: dict[str, float],
    target_torso_height: float | None = None,
    target_torso_pitch: float | None = None,
    target_torso_roll: float | None = None,
    target_torso_location_xy: tuple[float, float] | None = None,
    target_torso_velocity_xy: tuple[float, float] | None = None,
    target_torso_heading: float | None = None,
    target_turning_speed: float | None = None,
):
  """Updates cost_weights and task_parameters to specify torso targets.

  Arguments:
    cost_weights: a dict of weight values which will be mutated.
    task_parameters: a dict of task parameters which will be mutated.
    target_torso_height: how high the torso should be in meters.
    target_torso_pitch: pitch from a horizontal pose in radians. A positive
      number means robot is looking up.
    target_torso_roll: roll angle in radians, in clockwise direction
    target_torso_location_xy: a tuple containing target global coordinates in
      meters.
    target_torso_velocity_xy: target velocity in robot coordinates, where x is
      forward and y is right.
    target_torso_heading: desired global heading direction in radians.
    target_turning_speed: the desired turning speed of the torso in radians per
      second.
  """
  if target_torso_height is not None:
    task_parameters.update({
        'TargetBodyHeight': target_torso_height,
    })
    cost_weights.update({
        'BodyHeight': 0.5,
    })

  if target_torso_pitch is not None:
    task_parameters.update({
        'TargetPitch': -target_torso_pitch,
    })
    cost_weights.update({
        'Pitch': 0.6,
    })

  if target_torso_roll is not None:
    # Note: In Language2Reward paper, this was added to the current roll of the
    # robot.
    task_parameters.update({
        'TargetRoll': target_torso_roll,
    })
    cost_weights.update({
        'Roll': 1.0,
    })

  if target_torso_location_xy is not None:
    task_parameters.update({
        'GoalPositionX': target_torso_location_xy[0],
        'GoalPositionY': target_torso_location_xy[1],
    })
    cost_weights.update(
        {'BodyXYPos': 0.6, 'Forward_Velocity': 0.0, 'Turning_Velocity': 0.0}
    )
  else:
    cost_weights.update({'BodyXYPos': 0.0})

  if target_torso_velocity_xy is not None:
    task_parameters.update({
        'TargetForwardVelocity': target_torso_velocity_xy[0],
        'TargetSidewaysVelocity': target_torso_velocity_xy[1],
    })
    cost_weights.update({
        'BodyXYPos': 0.0,
        'Forward_Velocity': 0.1,
        'Sideways_Velocity': 0.1,
        'Turning_Velocity': 0.1,
    })
  else:
    cost_weights.update({
        'Forward_Velocity': 0.0,
        'Sideways_Velocity': 0.0,
        'Turning_Velocity': 0.0,
    })
  if target_torso_heading is not None:
    task_parameters.update({'TargetHeading': target_torso_heading})
    cost_weights.update({'Heading': 0.6, 'Turning_Velocity': 0.0})
  else:
    cost_weights.update({'Heading': 0.0})
  if target_turning_speed is not None:
    task_parameters.update({'TargetTurningVelocity': target_turning_speed})
    cost_weights.update({'Heading': 0.0, 'Turning_Velocity': 0.6})
  else:
    cost_weights.update({'Turning_Velocity': 0.0})


def set_foot_pos_parameters(
    cost_weights: dict[str, float],
    task_parameters: dict[str, float],
    foot_name: FootName,
    lift_height: float | None = None,
    extend_forward: float | None = None,
    move_inward: float | None = None,
):
  """Updates cost_weights and task_parameters to set the position of a foot."""
  foot_code_name = _FOOT_TO_CODENAME[foot_name]
  if lift_height is not None:
    balance = lift_height <= 0.0
    task_parameters.update({
        f'{foot_code_name}ZDistFromNominal': lift_height,
        f'{foot_code_name}Balance': balance,
    })
    cost_weights.update({f'{foot_code_name}_upward': 1.0})

  if extend_forward is not None:
    task_parameters.update(
        {f'{foot_code_name}XDistFromNominal': extend_forward}
    )
    cost_weights.update({f'{foot_code_name}_forward': 0.5})

  if move_inward is not None:
    task_parameters.update({f'{foot_code_name}YDistFromNominal': -move_inward})
    cost_weights.update({f'{foot_code_name}_inward': 0.5})


def set_foot_stepping_parameters(
    cost_weights: dict[str, float],
    task_parameters: dict[str, float],
    foot_name: FootName,
    stepping_frequency: float,
    air_ratio: float,
    phase_offset: float,
    swing_up_down: float,
    swing_forward_back: float,
    should_activate: bool,
):
  """Updates cost_weights and task_parameters to set gait parameters."""
  foot_code_name = _FOOT_TO_CODENAME[foot_name]
  if should_activate:
    task_parameters.update({
        f'{foot_code_name}SteppingFrequency': stepping_frequency,
        f'{foot_code_name}GroundToAirRatio': 1 - air_ratio,
        f'{foot_code_name}PhaseOffset': phase_offset,
        f'{foot_code_name}AmplitudesVertical': swing_up_down,
        f'{foot_code_name}AmplitudesForward': swing_forward_back,
    })
    cost_weights.update({
        f'{foot_code_name}_stepping_vertical': 1.0,
        f'{foot_code_name}_stepping_horizontal': 0.2,
    })
  else:
    cost_weights.update({
        f'{foot_code_name}_stepping_vertical': 0.0,
        f'{foot_code_name}_stepping_horizontal': 0.0,
    })


def set_target_joint_angles(
    cost_weights: dict[str, float],
    task_parameters: dict[str, float],
    leg_name: FootName,
    target_joint_angles,
):
  leg_id = _FOOT_TO_ID[leg_name]
  for i in range(3):
    task_parameters.update({f'q{leg_id*3+i+1}': target_joint_angles[i]})
  cost_weights.update({'PoseTracking': 1.0})


def head_towards(
    cost_weights: dict[str, float],
    task_parameters: dict[str, float],
    heading_direction: float,
):
  """Sets a global heading direction."""
  task_parameters['TargetHeading'] = heading_direction
  cost_weights['Heading'] = 0.3


def walk(
    cost_weights: dict[str, float],
    task_parameters: dict[str, float],
    forward_speed: float,
    sideways_speed: float,
    turning_speed: float,
):
  """Sets target velocities for the torso."""
  task_parameters.update({
      'TargetForwardVelocity': forward_speed,
      'TargetSidewaysVelocity': sideways_speed,
      'TargetTurningVelocity': turning_speed,
  })
  cost_weights.update({
      'BodyXYPos': 0.0,
      'Forward_Velocity': 0.1,
      'Sideways_Velocity': 0.1,
      'Turning_Velocity': 0.3,
  })


_FOOT_TO_CODENAME = {
    'front_left': 'FL',
    'rear_left': 'HL',
    'front_right': 'FR',
    'rear_right': 'HR',
    'back_left': 'HL',
    'back_right': 'HR',
}
_FOOT_TO_ID = {
    'front_left': 0,
    'rear_left': 1,
    'front_right': 2,
    'rear_right': 3,
    'back_left': 1,
    'back_right': 3,
}
