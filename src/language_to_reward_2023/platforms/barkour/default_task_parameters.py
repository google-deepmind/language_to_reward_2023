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

"""Parameters matching the default values in barkour_modeless.cc."""


def default_parameters_and_weights() -> (
    tuple[dict[str, float], dict[str, float]]
):
  """Returns parameters matching the default values in barkour_modeless.cc."""
  weights = {
      "Balance": 0.2,
      "BodyHeight": 0.0,
      "BodyXYPos": 0.0,
      "Effort": 0.0,
      "FL_forward": 0.0,
      "FL_inward": 0.0,
      "FL_stepping_horizontal": 0.0,
      "FL_stepping_vertical": 0.0,
      "FL_upward": 0.0,
      "FR_forward": 0.0,
      "FR_inward": 0.0,
      "FR_stepping_horizontal": 0.0,
      "FR_stepping_vertical": 0.0,
      "FR_upward": 0.0,
      "Forward_Velocity": 0.0,
      "HL_forward": 0.0,
      "HL_inward": 0.0,
      "HL_stepping_horizontal": 0.0,
      "HL_stepping_vertical": 0.0,
      "HL_upward": 0.0,
      "HR_forward": 0.0,
      "HR_inward": 0.0,
      "HR_stepping_horizontal": 0.0,
      "HR_stepping_vertical": 0.0,
      "HR_upward": 0.0,
      "Heading": 0.0,
      "Pitch": 0.0,
      "PoseTracking": 0.0,
      "Posture": 0.01,
      "Roll": 0.0,
      "Roll_Velocity": 0.0,
      "Sideways_Velocity": 0.0,
      "Turning_Velocity": 0.0,
      "Upward_Velocity": 0.0,
      "act_dot": 2e-05,
  }
  params = {
      "FLAmplitudesForward": 0.0,
      "FLAmplitudesVertical": 0.0,
      "FLBalance": 1.0,
      "FLGroundToAirRatio": 0.0,
      "FLPhaseOffset": 0.0,
      "FLSteppingFrequency": 0.0,
      "FLXDistFromNominal": 0.0,
      "FLYDistFromNominal": 0.0,
      "FLZDistFromNominal": 0.0,
      "FRAmplitudesForward": 0.0,
      "FRAmplitudesVertical": 0.0,
      "FRBalance": 1.0,
      "FRGroundToAirRatio": 0.0,
      "FRPhaseOffset": 0.0,
      "FRSteppingFrequency": 0.0,
      "FRXDistFromNominal": 0.0,
      "FRYDistFromNominal": 0.0,
      "FRZDistFromNominal": 0.0,
      "GoalPositionX": 0.0,
      "GoalPositionY": 0.0,
      "HLAmplitudesForward": 0.0,
      "HLAmplitudesVertical": 0.0,
      "HLBalance": 1.0,
      "HLGroundToAirRatio": 0.0,
      "HLPhaseOffset": 0.0,
      "HLSteppingFrequency": 0.0,
      "HLXDistFromNominal": 0.0,
      "HLYDistFromNominal": 0.0,
      "HLZDistFromNominal": 0.0,
      "HRAmplitudesForward": 0.0,
      "HRAmplitudesVertical": 0.0,
      "HRBalance": 1.0,
      "HRGroundToAirRatio": 0.0,
      "HRPhaseOffset": 0.0,
      "HRSteppingFrequency": 0.0,
      "HRXDistFromNominal": 0.0,
      "HRYDistFromNominal": 0.0,
      "HRZDistFromNominal": 0.0,
      "TargetBodyHeight": 0.292,
      "TargetForwardVelocity": 0.0,
      "TargetHeading": 0.0,
      "TargetPitch": 0.0,
      "TargetRoll": 0.0,
      "TargetRollVelocity": 0.0,
      "TargetSidewaysVelocity": 0.0,
      "TargetTurningVelocity": 0.0,
      "TargetUpwardVelocity": 0.0,
      "q1": 0.0,
      "q10": 0.0,
      "q11": 0.0,
      "q12": 0.0,
      "q2": 0.0,
      "q3": 0.0,
      "q4": 0.0,
      "q5": 0.0,
      "q6": 0.0,
      "q7": 0.0,
      "q8": 0.0,
      "q9": 0.0,
  }
  return weights, params
