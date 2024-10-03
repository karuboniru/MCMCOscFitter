#pragma once

#include "StateI.h"
#include <TH2.h>
#include <array>

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

  [[nodiscard]] virtual std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta) const = 0;

  [[nodiscard, gnu::pure]] auto GetDM2() const {
    return is_NH ? NH_DM2 : IH_DM2;
  }
  [[nodiscard, gnu::pure]] auto GetT23() const {
    return is_NH ? NH_T23 : IH_T23;
  }
  [[nodiscard, gnu::pure]] auto GetT13() const {
    return is_NH ? NH_T13 : IH_T13;
  }
  [[nodiscard, gnu::pure]] auto GetDm2() const {
    return is_NH ? NH_Dm2 : IH_Dm2;
  }
  [[nodiscard, gnu::pure]] auto GetT12() const {
    return is_NH ? NH_T12 : IH_T12;
  }
  [[nodiscard, gnu::pure]] auto GetDeltaCP() const {
    return is_NH ? NH_DCP : IH_DCP;
  }

private:
  // PDG Central Values
  static constexpr double DM2 = 2.453e-3;
  static constexpr double DM2_IH = -2.536e-3;
  static constexpr double Theta23 = 0.546;
  static constexpr double Theta23_IH = 0.539;
  static constexpr double Theta13 = 0.022;
  static constexpr double dm2 = 7.53e-5;
  static constexpr double Theta12 = 0.307;

  static constexpr double sigma_t12 = 0.013;
  static constexpr double sigma_t13 = 0.07e-2;
  static constexpr double sigma_dm2 = 0.18e-5;
  static constexpr double sigma_t23 = 0.021;
  static constexpr double sigma_t23_IH = 0.022;
  static constexpr double sigma_DM2 = 0.033e-3;
  static constexpr double sigma_DM2_IH = 0.034e-3;

  // current state
  // private:
  bool is_NH{true};
  double NH_DM2{DM2}, NH_T23{Theta23}, NH_T13{Theta13}, NH_Dm2{dm2},
      NH_T12{Theta12}, NH_DCP{0.};
  double IH_DM2{DM2_IH}, IH_T23{Theta23_IH}, IH_T13{Theta13}, IH_Dm2{dm2},
      IH_T12{Theta12}, IH_DCP{0.};
};
