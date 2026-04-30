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

  // IHistogramPropagator overrides — delegate to the const overloads above.
  [[nodiscard]] std::array<std::array<std::array<TH2D, 2>, 2>, 2>
  GetProb_Hists(const std::vector<double> &Ebin,
                const std::vector<double> &costhbin,
                const OscillationParameters &p) override {
    return static_cast<const Prob3ppOscillation *>(this)->GetProb_Hists(
        Ebin, costhbin, p);
  }

  [[nodiscard]] std::array<std::array<std::array<TH2D, 3>, 3>, 2>
  GetProb_Hists_3F(const std::vector<double> &Ebin,
                   const std::vector<double> &costhbin,
                   const OscillationParameters &p) override {
    return static_cast<const Prob3ppOscillation *>(this)->GetProb_Hists_3F(
        Ebin, costhbin, p);
  }

  [[nodiscard]] std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta,
          const OscillationParameters &p) const;

  void re_calculate(const OscillationParameters &p) override;

  // void re_calculate() override {}
};