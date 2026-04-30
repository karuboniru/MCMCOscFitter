#pragma once

#include "IHistogramPropagator.h"
#include "OscillationParameters.h"
class OscillationParameters;

class Prob3ppOscillation : public IHistogramPropagator {
public:
  Prob3ppOscillation() = default;
  Prob3ppOscillation(const Prob3ppOscillation &) = default;
  Prob3ppOscillation(Prob3ppOscillation &&) noexcept = default;
  Prob3ppOscillation &operator=(const Prob3ppOscillation &) = default;
  Prob3ppOscillation &operator=(Prob3ppOscillation &&) noexcept = default;
  ~Prob3ppOscillation() = default;

  template <typename... Args>
  Prob3ppOscillation(Args &&...) {} // dummy constructor

  [[nodiscard]] std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta,
          const OscillationParameters &p) const;

  [[nodiscard]] std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2>
  GetProb_Hists_POD(const std::vector<double> &Ebin,
                    const std::vector<double> &costhbin,
                    const OscillationParameters &p) override;

  [[nodiscard]] std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2>
  GetProb_Hists_3F_POD(const std::vector<double> &Ebin,
                       const std::vector<double> &costhbin,
                       const OscillationParameters &p) override;

  void re_calculate(const OscillationParameters &p) override;
};
