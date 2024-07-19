#pragma once

#include "StateI.h"
#include <array>

class OscillationParameters : virtual public StateI {
public:
  OscillationParameters() = default;
  OscillationParameters(const OscillationParameters &) = default;
  OscillationParameters(OscillationParameters &&) = default;
  OscillationParameters &operator=(const OscillationParameters &) = default;
  OscillationParameters &operator=(OscillationParameters &&) = default;

  virtual ~OscillationParameters() = default;

  virtual void proposeStep() override;
  virtual double GetLogLikelihood() const override;

  virtual std::array<std::array<double, 3>, 3> GetProb(int flavor, double E,
                                                       double costheta) const = 0;

  double GetDM2() const { return current_DM2; }
  double GetT23() const { return current_Theta23; }

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
protected:
  double current_DM2{DM2}, current_Theta23{Theta23}, current_Theta13{Theta13},
      current_dm2{dm2}, current_Theta12{Theta12};
};

class Prob3ppOscillation : public OscillationParameters {
  public:
  Prob3ppOscillation() : OscillationParameters() {}
  virtual std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta) const override;
};