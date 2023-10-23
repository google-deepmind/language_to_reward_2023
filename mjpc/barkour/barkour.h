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

#ifndef THIRD_PARTY_DEEPMIND_LANGUAGE_TO_REWARD_2023_MJPC_BARKOUR_BARKOUR_H_
#define THIRD_PARTY_DEEPMIND_LANGUAGE_TO_REWARD_2023_MJPC_BARKOUR_BARKOUR_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include <mjpc/task.h>


namespace mjpc::language2reward {

class Barkour : public mjpc::Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Barkour* task) :
        mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Barkour;
    //  ============  enums  ============

    // feet
    enum BkFoot { kFootFL = 0, kFootHL, kFootFR, kFootHR, kNumFoot };

    //  ============  constants  ============
    constexpr static BkFoot kFootAll[kNumFoot] = {kFootFL, kFootHL, kFootFR,
                                                  kFootHR};

    // posture gain factors for abduction, hip, knee
    constexpr static double kJointPostureGain[3] = {2, 1, 1};  // unitless

    constexpr static char kRootJointName[] = "torso";

    //  ============  methods  ============

    // compute average foot position, depending on mode
    void AverageFootPos(double avg_foot_pos[3],
                        double* foot_pos[kNumFoot]) const;

    void GetNormalizedFootTrajectory(
        double duty_ratio, double gait_frequency,
        double normalized_phase_offset, double time,
        double amplitude_forward, double amplitude_vertical,
        double* target_pos_x, double* target_pos_z) const;

    void ResetLocked(const mjModel* model);

    //  ============  constants, computed in Reset()  ============
    int torso_body_id_ = -1;

    // ===== task parameters id ======
    int target_body_height_id_ = -1;
    int goal_position_x_id_ = -1;
    int goal_position_y_id_ = -1;
    int target_body_heading_id_ = -1;
    int target_body_pitch_id_ = -1;
    int target_body_roll_id_ = -1;
    int target_forward_velocity_id_ = -1;
    int target_sideways_velocity_id_ = -1;
    int target_upward_velocity_id_ = -1;
    int target_turning_velocity_id_ = -1;
    int target_roll_velocity_id_ = -1;
    int balance_ids_[4] = {-1};

    // ==== gait-relevant parameters id ====
    int dist_from_nominal_ids_[4][3];
    int stepping_frequency_ids_[4];
    int ground_to_air_ratio_ids_[4];
    int phase_offset_ids_[4];
    int amplitudes_vertical_ids_[4];
    int amplitudes_forward_ids_[4];

    // ==== joint angle parameters id ====
    int target_joint_angle_ids_[12];

    // ==== object root joint ids ====
    int walker_root_joint_id_ = -1;

    double current_base_roll_ = 0;

    int foot_site_id_[kNumFoot];
    int shoulder_body_id_[kNumFoot];
  };

  Barkour() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

  // call base-class Reset, save task-related ids
  void ResetLocked(const mjModel* model) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  int goal_mocap_id_ = -1;
};

}  // namespace mjpc::language2reward

#endif  // THIRD_PARTY_DEEPMIND_LANGUAGE_TO_REWARD_2023_MJPC_BARKOUR_BARKOUR_H_
