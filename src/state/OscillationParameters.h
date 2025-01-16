#pragma once

#include "StateI.h"
#include <TH2.h>
#include <array>

#include <cmath>
#include <iostream>

struct param {
  double DM2;
  double Dm2;
  double T23;
  double T13;
  double T12;
  double DCP;
};

class pull_toggle {
public:
  std::array<bool, 6> flags{};
  constexpr static auto names =
      std::to_array<std::string>({"DM32", "DM21", "T23", "T13", "T12", "DCP"});
  [[nodiscard]] std::vector<std::string> get_active() const {
    std::vector<std::string> active;
    for (size_t i = 0; i < flags.size(); ++i) {
      if (flags[i]) {
        active.push_back(names[i]);
      }
    }
    return active;
  }
  [[nodiscard]] std::vector<std::string> get_inactive() const {
    std::vector<std::string> inactive;
    for (size_t i = 0; i < flags.size(); ++i) {
      if (!flags[i]) {
        inactive.push_back(names[i]);
      }
    }
    return inactive;
  }
  bool operator[](size_t i) const { return flags[i]; }
  bool &operator[](size_t i) { return flags[i]; }
};

constexpr pull_toggle all_on{.flags = {true, true, true, true, true, true}};
constexpr pull_toggle SK_w_T13{.flags = {false, true, false, true, true, false}};
constexpr pull_toggle SK_wo_T13{.flags = {false, true, false, false, true, false}};
constexpr pull_toggle all_off{
    .flags = {false, false, false, false, false, false}};

class OscillationParameters : virtual public StateI {
public:
  OscillationParameters() = default;
  OscillationParameters(const OscillationParameters &) = default;
  OscillationParameters(OscillationParameters &&) = default;
  OscillationParameters &operator=(const OscillationParameters &) = default;
  OscillationParameters &operator=(OscillationParameters &&) = default;

  ~OscillationParameters() override = default;

  void proposeStep() override;
  [[nodiscard]] double GetLogLikelihood() const override;
  [[nodiscard]] double GetLogLikelihood(const pull_toggle &) const;

  [[nodiscard]] virtual std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta) const = 0;

  [[nodiscard, gnu::pure]] auto GetDM32sq() const {
    return is_NH ? NH_DM2 : IH_DM2;
  }
  [[nodiscard, gnu::pure]] auto GetT23() const {
    return is_NH ? NH_T23 : IH_T23;
  }
  [[nodiscard, gnu::pure]] auto GetT13() const {
    return is_NH ? NH_T13 : IH_T13;
  }
  [[nodiscard, gnu::pure]] auto GetDM21sq() const {
    return is_NH ? NH_Dm2 : IH_Dm2;
  }
  [[nodiscard, gnu::pure]] auto GetT12() const {
    return is_NH ? NH_T12 : IH_T12;
  }
  [[nodiscard, gnu::pure]] auto GetDeltaCP() const {
    return is_NH ? NH_DCP : IH_DCP;
  }

  void flip_hierarchy() { is_NH = !is_NH; }

  void set_param(const param &p);

  virtual void re_calculate() = 0;

  void set_toggle(const pull_toggle & new_toggle){
    current_toggle = new_toggle;
  }

private:
  // PDG Central Values
  static constexpr double DM2 = 2.455e-3;
  static constexpr double DM2_IH = -2.529e-3;
  static constexpr double Theta23 = 0.558;
  static constexpr double Theta23_IH = 0.553;
  static constexpr double Theta13 = 2.19e-2;
  static constexpr double dm2 = 7.53e-5;
  static constexpr double Theta12 = 0.307;
  static constexpr double DCP = 1.19 * M_PI;

  static constexpr double sigma_t12 = 0.013;
  static constexpr double sigma_t13 = 0.07e-2;
  static constexpr double sigma_dm2 = 0.18e-5;

  static constexpr double sigma_t23_down = 0.021;
  static constexpr double sigma_t23_up = 0.015;
  static constexpr double sigma_t23 = (sigma_t23_down + sigma_t23_up) / 2;

  static constexpr double sigma_t23_IH_down = 0.024;
  static constexpr double sigma_t23_IH_up = 0.016;
  static constexpr double sigma_t23_IH =
      (sigma_t23_IH_down + sigma_t23_IH_up) / 2;
  static constexpr double sigma_DM2 = 0.028e-3;
  static constexpr double sigma_DM2_IH = 0.029e-3;
  static constexpr double sigma_DCP = 0.22 * M_PI;

  bool is_NH{true};
  double NH_DM2{DM2}, NH_T23{Theta23}, NH_T13{Theta13}, NH_Dm2{dm2},
      NH_T12{Theta12}, NH_DCP{DCP};
  double IH_DM2{DM2_IH}, IH_T23{Theta23_IH}, IH_T13{Theta13}, IH_Dm2{dm2},
      IH_T12{Theta12}, IH_DCP{DCP};
  pull_toggle current_toggle{all_on};
};
