// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "mjpc/barkour/barkour.h"

#include <cmath>
#include <string>
#include <string_view>

#include <absl/log/check.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include <mjpc/utilities.h>

namespace mjpc::language2reward {

namespace {  // anonymous namespace for local definitions

constexpr double kNominalFootFromShoulder[4][2] = {{0.05, 0.0},
                                                   {-0.05, 0.0},
                                                   {0.05, -0.0},
                                                   {-0.05, -0.0}};

//  ============  reusable utilities  ============

void RotateVectorAroundZaxis(const double* vec, const double angle,
                             double* rotated_vec) {
  rotated_vec[0] = mju_cos(angle) * vec[0] - mju_sin(angle) * vec[1];
  rotated_vec[1] = mju_sin(angle) * vec[0] + mju_cos(angle) * vec[1];
  rotated_vec[2] = vec[2];
}

// Unwind the roll current angle to be close to last measured roll angle
// to prevent sudden jump of the measured sensor data.
double UnwindAngle(const double current_roll, const double last_roll) {
  double vec[3] = {current_roll - 2 * mjPI, current_roll,
                   current_roll + 2 * mjPI};
  double closest = vec[0];
  double min_diff = std::abs(last_roll - vec[0]);
  for (int i = 1; i < 3; i++) {
    double diff = std::abs(last_roll - vec[i]);
    if (diff < min_diff) {
      min_diff = diff;
      closest = vec[i];
    }
  }
  return closest;
}

}  // namespace

std::string Barkour::XmlPath() const {
  return absl::StrCat(
      mjpc::GetExecutableDir(), "/../mjpc/barkour/world.xml");
}

std::string Barkour::Name() const { return "Barkour"; }

void Barkour::ResidualFn::GetNormalizedFootTrajectory(
    double duty_ratio, double gait_frequency, double normalized_phase_offset,
    double time, double amplitude_forward, double amplitude_vertical,
    double* target_pos_x, double* target_pos_z) const {
  double step_duration = 1.0 / fmax(gait_frequency, 0.0001);
  double stance_duration = step_duration * fmin(fmax(duty_ratio, 0.01), 0.99);
  double flight_duration = step_duration - stance_duration;
  double gait_start_time = normalized_phase_offset * step_duration;
  double in_phase_time = fmod((time - gait_start_time), step_duration);
  if (in_phase_time < 0.0) in_phase_time += step_duration;
  double normalized_phase = 0.0;
  bool is_flight_phase = false;
  if (in_phase_time < flight_duration) {
    normalized_phase = in_phase_time / flight_duration;
    is_flight_phase = true;
  } else {
    normalized_phase = (in_phase_time - flight_duration) / stance_duration;
    is_flight_phase = false;
  }
  if (is_flight_phase) {
    *target_pos_z += mju_sin(normalized_phase * mjPI) * amplitude_vertical;
    *target_pos_x +=
        mju_sin(normalized_phase * mjPI - mjPI / 2) * amplitude_forward;
  } else {
    *target_pos_x +=
        -mju_sin(normalized_phase * mjPI - mjPI / 2) * amplitude_forward;
  }
}

//  ============  residual  ============
void Barkour::ResidualFn::Residual(const mjModel* model,
                                            const mjData* data,
                                            double* residual) const {
  // start counter
  int counter = 0;

  // common info
  double* torso_xmat = data->xmat + 9 * torso_body_id_;
  double* compos = SensorByName(model, data, "torso_subtreecom");
  CHECK(compos != nullptr);
  double pitch = atan2(-torso_xmat[6], mju_sqrt(torso_xmat[7] * torso_xmat[7] +
                                                torso_xmat[8] * torso_xmat[8]));
  double roll = atan2(torso_xmat[7], torso_xmat[8]);

  double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
  if (parameters_[target_body_pitch_id_] > 1.0 ||
      parameters_[target_body_pitch_id_] < -1.0) {
    int handstand = parameters_[target_body_pitch_id_] < 0.0 ? -1 : 1;
    torso_heading[0] = handstand * torso_xmat[2];
    torso_heading[1] = handstand * torso_xmat[5];
  }

  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double* angvel = SensorByName(model, data, "torso_angvel");
  double linvel_ego[3];
  RotateVectorAroundZaxis(comvel,
                          -mju_atan2(torso_heading[1], torso_heading[0]),
                          linvel_ego);  // Need double check

  // get foot positions
  double* foot_pos[kNumFoot];
  for (BkFoot foot : kFootAll) {
    foot_pos[foot] = data->site_xpos + 3 * foot_site_id_[foot];
  }

  double* shoulder_pos[kNumFoot];
  for (BkFoot foot : kFootAll)
    shoulder_pos[foot] = data->xpos + 3 * shoulder_body_id_[foot];

  // average foot position
  double avg_foot_pos[3];
  AverageFootPos(avg_foot_pos, foot_pos);

  // Residual Calculations
  // ---------- BodyHeight ----------
  // quadrupedal or bipedal height of torso
  double* body = SensorByName(model, data, "body");
  residual[counter++] =
      body[2] - parameters_[target_body_height_id_];

  // ---------- BodyXYPos ----------
  double* pos = SensorByName(model, data, "torso_subtreecom");
  CHECK(pos != nullptr);
  residual[counter++] = pos[0] - parameters_[goal_position_x_id_];
  residual[counter++] = pos[1] - parameters_[goal_position_y_id_];

  // ---------- Heading ----------
  mju_normalize(torso_heading, 2);
  residual[counter++] =
      torso_heading[0] - mju_cos(parameters_[target_body_heading_id_]);
  residual[counter++] =
      torso_heading[1] - mju_sin(parameters_[target_body_heading_id_]);

  // ---------- Body Roll and Pitch ----------
  residual[counter++] = pitch - parameters_[target_body_pitch_id_];
  residual[counter++] =
      UnwindAngle(roll, current_base_roll_) - parameters_[target_body_roll_id_];
  residual[counter++] = 0;

  // ---------- Forward Velocity ----------
  residual[counter++] =
      linvel_ego[0] - parameters_[target_forward_velocity_id_];

  // ---------- Sideways Velocity ---------
  residual[counter++] =
      linvel_ego[1] - parameters_[target_sideways_velocity_id_];

  // ---------- Upward Velocity -----------
  residual[counter++] = linvel_ego[2] - parameters_[target_upward_velocity_id_];

  // ---------- Roll Pitch Yaw Velocity ----------
  residual[counter++] = angvel[0] - parameters_[target_roll_velocity_id_];
  residual[counter++] = angvel[2] - parameters_[target_turning_velocity_id_];

  // ---------- Feet pose tracking -------------
  double foot_current_poses_yaw_aligned[4][3];
  double shoulder_yaw_aligned[4][3];
  double heading_angle = mju_atan2(torso_heading[1], torso_heading[0]);

  for (int foot_id = 0; foot_id < 4; foot_id++) {
    double shoulder_pos_local[3];
    mju_sub3(shoulder_pos_local, shoulder_pos[foot_id], compos);
    RotateVectorAroundZaxis(shoulder_pos_local, -heading_angle,
                            shoulder_yaw_aligned[foot_id]);
    double foot_pos_local[3];
    mju_sub3(foot_pos_local, foot_pos[foot_id], compos);
    RotateVectorAroundZaxis(foot_pos_local, -heading_angle,
                            foot_current_poses_yaw_aligned[foot_id]);
    foot_current_poses_yaw_aligned[foot_id][2] = foot_pos[foot_id][2];
  }
  for (int dim = 0; dim < 3; dim++) {
    for (int foot_id = 0; foot_id < 4; foot_id++) {
      double foot_pos_delta = parameters_[dist_from_nominal_ids_[foot_id][dim]];
      double foot_target_pos;
      if (dim < 2) {
        if (foot_id >= 2 && dim == 1) {
          foot_pos_delta *= -1;
        }
        foot_target_pos = shoulder_yaw_aligned[foot_id][dim] +
                          kNominalFootFromShoulder[foot_id][dim] +
                          foot_pos_delta;
      } else {
        // dim == 2
        foot_target_pos = foot_pos_delta;
      }
      residual[counter++] =
          foot_target_pos - foot_current_poses_yaw_aligned[foot_id][dim];
    }
  }

  // ---------- Gait ----------
  // Get tracking target for the feet gait patter
  for (int foot_id = 0; foot_id < 4; foot_id++) {
    double target_pos_x = shoulder_yaw_aligned[foot_id][0] +
                          kNominalFootFromShoulder[foot_id][0] +
                          parameters_[dist_from_nominal_ids_[foot_id][0]];
    double target_pos_z = parameters_[dist_from_nominal_ids_[foot_id][2]];
    GetNormalizedFootTrajectory(parameters_[ground_to_air_ratio_ids_[foot_id]],
                                parameters_[stepping_frequency_ids_[foot_id]],
                                parameters_[phase_offset_ids_[foot_id]],
                                data->time,
                                parameters_[amplitudes_forward_ids_[foot_id]],
                                parameters_[amplitudes_vertical_ids_[foot_id]],
                                &target_pos_x, &target_pos_z);
    residual[counter++] =
        target_pos_z - foot_current_poses_yaw_aligned[foot_id][2];
    residual[counter++] =
        target_pos_x - foot_current_poses_yaw_aligned[foot_id][0];
  }

  // ---------- Balance ----------
  double capture_point[3];
  double fall_time = mju_sqrt(2 * parameters_[target_body_height_id_] / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];

  // ---------- Effort ----------
  mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
  counter += model->nu;

  // ---------- Posture ----------
  double* home = KeyQPosByName(model, data, "home");
  int walker_first_joint_adr = model->jnt_qposadr[walker_root_joint_id_] + 7;
  mju_sub(residual + counter,
          data->qpos + walker_first_joint_adr,
          home + walker_first_joint_adr, model->nu);
  for (BkFoot foot : kFootAll) {
    for (int joint = 0; joint < 3; joint++) {
      residual[counter + 3 * foot + joint] *= kJointPostureGain[joint];
    }
  }
  // loosen the "hands" in Biped mode
  if (parameters_[target_body_pitch_id_] < -1.0) {
    residual[counter + 4] *= 0.03;
    residual[counter + 5] *= 0.03;
    residual[counter + 10] *= 0.03;
    residual[counter + 11] *= 0.03;
  } else if (parameters_[target_body_pitch_id_] > 1.0) {
    residual[counter + 1] *= 0.03;
    residual[counter + 2] *= 0.03;
    residual[counter + 7] *= 0.03;
    residual[counter + 8] *= 0.03;
  }
  counter += model->nu;

  // PoseTracking
  for (int i = 0; i < 12; i++) {
    residual[counter + i] =
        parameters_[target_joint_angle_ids_[i]] -
        *(data->qpos + walker_first_joint_adr + i);
  }
  counter += model->nu;

  // ---------- Act dot -----------------
  // encourage actions to be similar to the previous actions.
  if (model->na > 0) {
    mju_copy(residual + counter, data->act_dot, model->na);
    counter += model->na;
  } else {
    for (int i=0; i < 12; i++) {
      residual[counter++] = 0.;
    }
  }

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

//  ============  transition  ============
void Barkour::TransitionLocked(mjModel* model, mjData* data) {
  // Update goal position for mocap object
  data->mocap_pos[3 * goal_mocap_id_] =
      parameters[residual_.goal_position_x_id_];
  data->mocap_pos[3 * goal_mocap_id_ + 1] =
      parameters[residual_.goal_position_y_id_];

  // update the body roll variable
  double* torso_xmat = data->xmat + 9 * residual_.torso_body_id_;
  // xmat[7,8] are the projections of the local y and z axes onto the global
  // z axis, respectively.
  double roll_angle = atan2(torso_xmat[7], torso_xmat[8]);
  // This is needed to handle when the roll_angle jumps between PI and -PI,
  // which creates undesired discontinuity in the optimization.
  residual_.current_base_roll_ =
      UnwindAngle(roll_angle, residual_.current_base_roll_);
}

//  ============  task-state utilities  ============
// save task-related ids
void Barkour::ResetLocked(const mjModel* model) {
  const int* mocap_id = model->body_mocapid;
  goal_mocap_id_ = mocap_id[mj_name2id(model, mjOBJ_XBODY, "goal")];
  residual_.ResetLocked(model);
}

void Barkour::ResidualFn::ResetLocked(const mjModel* model) {
  // ----------  model identifiers  ----------
  torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "torso");
  CHECK_GT(torso_body_id_, 0);
  // foot site ids
  int foot_index = 0;
  // for (const char* footname : {"FL", "HL", "FR", "HR"}) {
  for (const char* footname :
       {"foot_front_left", "foot_hind_left",
        "foot_front_right", "foot_hind_right"}) {
    int foot_id = mj_name2id(model, mjOBJ_SITE, footname);
    if (foot_id < 0) mju_error_s("site '%s' not found", footname);
    foot_site_id_[foot_index] = foot_id;
    foot_index++;
  }

  // shoulder ids
  shoulder_body_id_[kFootFL] =
      mj_name2id(model, mjOBJ_BODY, "upper_leg_front_left");
  shoulder_body_id_[kFootHL] =
      mj_name2id(model, mjOBJ_BODY, "upper_leg_hind_left");
  shoulder_body_id_[kFootFR] =
      mj_name2id(model, mjOBJ_BODY, "upper_leg_front_right");
  shoulder_body_id_[kFootHR] =
      mj_name2id(model, mjOBJ_BODY, "upper_leg_hind_right");
  CHECK_GT(shoulder_body_id_[kFootFL], 0);
  CHECK_GT(shoulder_body_id_[kFootHL], 0);
  CHECK_GT(shoulder_body_id_[kFootFR], 0);
  CHECK_GT(shoulder_body_id_[kFootHR], 0);

  // get the reward parameter_ids
  target_body_height_id_ = ParameterIndex(model, "TargetBodyHeight");
  goal_position_x_id_ = ParameterIndex(model, "GoalPositionX");
  goal_position_y_id_ = ParameterIndex(model, "GoalPositionY");
  target_body_heading_id_ = ParameterIndex(model, "TargetHeading");
  target_body_pitch_id_ = ParameterIndex(model, "TargetPitch");
  target_body_roll_id_ = ParameterIndex(model, "TargetRoll");
  target_forward_velocity_id_ = ParameterIndex(model, "TargetForwardVelocity");
  target_sideways_velocity_id_ =
      ParameterIndex(model, "TargetSidewaysVelocity");
  target_upward_velocity_id_ = ParameterIndex(model, "TargetUpwardVelocity");
  target_turning_velocity_id_ = ParameterIndex(model, "TargetTurningVelocity");
  target_roll_velocity_id_ = ParameterIndex(model, "TargetRollVelocity");

  balance_ids_[kFootFL] = ParameterIndex(model, "FLBalance");
  CHECK_GE(balance_ids_[kFootFL], 0);
  balance_ids_[kFootHL] = ParameterIndex(model, "HLBalance");
  CHECK_GE(balance_ids_[kFootHL], 0);
  balance_ids_[kFootFR] = ParameterIndex(model, "FRBalance");
  CHECK_GE(balance_ids_[kFootFR], 0);
  balance_ids_[kFootHR] = ParameterIndex(model, "HRBalance");
  CHECK_GE(balance_ids_[kFootHR], 0);

  dist_from_nominal_ids_[0][0] = ParameterIndex(model, "FLXDistFromNominal");
  dist_from_nominal_ids_[0][1] = ParameterIndex(model, "FLYDistFromNominal");
  dist_from_nominal_ids_[0][2] = ParameterIndex(model, "FLZDistFromNominal");
  stepping_frequency_ids_[0] = ParameterIndex(model, "FLSteppingFrequency");
  ground_to_air_ratio_ids_[0] = ParameterIndex(model, "FLGroundToAirRatio");
  phase_offset_ids_[0] = ParameterIndex(model, "FLPhaseOffset");
  amplitudes_vertical_ids_[0] = ParameterIndex(model, "FLAmplitudesVertical");
  amplitudes_forward_ids_[0] = ParameterIndex(model, "FLAmplitudesForward");
  dist_from_nominal_ids_[1][0] = ParameterIndex(model, "HLXDistFromNominal");
  dist_from_nominal_ids_[1][1] = ParameterIndex(model, "HLYDistFromNominal");
  dist_from_nominal_ids_[1][2] = ParameterIndex(model, "HLZDistFromNominal");
  stepping_frequency_ids_[1] = ParameterIndex(model, "HLSteppingFrequency");
  ground_to_air_ratio_ids_[1] = ParameterIndex(model, "HLGroundToAirRatio");
  phase_offset_ids_[1] = ParameterIndex(model, "HLPhaseOffset");
  amplitudes_vertical_ids_[1] = ParameterIndex(model, "HLAmplitudesVertical");
  amplitudes_forward_ids_[1] = ParameterIndex(model, "HLAmplitudesForward");
  dist_from_nominal_ids_[2][0] = ParameterIndex(model, "FRXDistFromNominal");
  dist_from_nominal_ids_[2][1] = ParameterIndex(model, "FRYDistFromNominal");
  dist_from_nominal_ids_[2][2] = ParameterIndex(model, "FRZDistFromNominal");
  stepping_frequency_ids_[2] = ParameterIndex(model, "FRSteppingFrequency");
  ground_to_air_ratio_ids_[2] = ParameterIndex(model, "FRGroundToAirRatio");
  phase_offset_ids_[2] = ParameterIndex(model, "FRPhaseOffset");
  amplitudes_vertical_ids_[2] = ParameterIndex(model, "FRAmplitudesVertical");
  amplitudes_forward_ids_[2] = ParameterIndex(model, "FRAmplitudesForward");
  dist_from_nominal_ids_[3][0] = ParameterIndex(model, "HRXDistFromNominal");
  dist_from_nominal_ids_[3][1] = ParameterIndex(model, "HRYDistFromNominal");
  dist_from_nominal_ids_[3][2] = ParameterIndex(model, "HRZDistFromNominal");
  stepping_frequency_ids_[3] = ParameterIndex(model, "HRSteppingFrequency");
  ground_to_air_ratio_ids_[3] = ParameterIndex(model, "HRGroundToAirRatio");
  phase_offset_ids_[3] = ParameterIndex(model, "HRPhaseOffset");
  amplitudes_vertical_ids_[3] = ParameterIndex(model, "HRAmplitudesVertical");
  amplitudes_forward_ids_[3] = ParameterIndex(model, "HRAmplitudesForward");

  for (int i = 0; i < 12; i++) {
    target_joint_angle_ids_[i] = ParameterIndex(model, absl::StrCat("q", i+1));
  }

  walker_root_joint_id_ = mj_name2id(model, mjOBJ_JOINT, kRootJointName);
  if (walker_root_joint_id_ == -1) {
    mju_error("root joint '%s' not found", kRootJointName);
  }
}

// compute average foot position, depending on mode
void Barkour::ResidualFn::AverageFootPos(
    double avg_foot_pos[3], double* foot_pos[kNumFoot]) const {
  // we should compute the average foot position for feet that are used for
  // balancing.
  // start by using all feet
  double foot_weight[kNumFoot] = {};
  double total_weight = 0;
  for (int foot : kFootAll) {
    foot_weight[foot] = parameters_[balance_ids_[foot]];
    total_weight += foot_weight[foot];
  }
  if (total_weight == 0) {
    // if no feet are considered, just use the average
    for (int foot : kFootAll) {
      foot_weight[foot] = 1.0;
    }
    total_weight = 4.0;
  }

  mju_zero3(avg_foot_pos);
  for (int foot : kFootAll) {
    mju_addToScl3(avg_foot_pos, foot_pos[foot], foot_weight[foot]);
  }
  mju_scl3(avg_foot_pos, avg_foot_pos, 1 / total_weight);
}

}  // namespace mjpc::language2reward
