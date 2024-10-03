#pragma once

#include "OscillationParameters.h"

class Prob3ppOscillation : public OscillationParameters {
public:
  Prob3ppOscillation() = default;
  [[nodiscard]] std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta) const override;

  [[nodiscard]] std::array<std::array<TH2D, 2>, 2>
  GetProb_Hist(std::vector<double> Ebin, std::vector<double> costhbin,
               int flavor) const;
};