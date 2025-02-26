#pragma once

#include "OscillationParameters.h"
class OscillationParameters;

class Prob3ppOscillation {
public:
  Prob3ppOscillation() = default;
  Prob3ppOscillation(const Prob3ppOscillation &) = default;
  Prob3ppOscillation(Prob3ppOscillation &&) noexcept = default;
  Prob3ppOscillation &operator=(const Prob3ppOscillation &) = default;
  Prob3ppOscillation &operator=(Prob3ppOscillation &&) noexcept = default;
  ~Prob3ppOscillation() = default;

  template <typename... Args>
  Prob3ppOscillation(Args &&...) {} // dummy constructor

  // [[nodiscard]] std::array<std::array<double, 3>, 3>
  // GetProb(int flavor, double E, double costheta) const override;

  [[nodiscard]] std::array<std::array<TH2D, 2>, 2>
  GetProb_Hist(std::vector<double> Ebin, std::vector<double> costhbin,
               int flavor, const OscillationParameters &p) const;
  [[nodiscard]] std::array<std::array<TH2D, 3>, 3>
  GetProb_Hist_3F(std::vector<double> Ebin, std::vector<double> costhbin,
                  int flavor, const OscillationParameters &p) const;

  [[nodiscard]] std::array<std::array<std::array<TH2D, 2>, 2>, 2>
  GetProb_Hists(std::vector<double> Ebin, std::vector<double> costhbin,
                const OscillationParameters &p) const;

  [[nodiscard]] std::array<std::array<std::array<TH2D, 3>, 3>, 2>
  GetProb_Hists_3F(std::vector<double> Ebin, std::vector<double> costhbin,
                   const OscillationParameters &p) const;

  [[nodiscard]] std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta,
          const OscillationParameters &p) const;

  // void re_calculate() override {}
};