#pragma once

#include "OscillationParameters.h"
#include <TH2.h>
#include <array>
#include <vector>

// Abstract interface for oscillation probability calculators that produce
// binned 2D probability histograms (E vs cos-theta).
// Both Prob3ppOscillation and ParProb3ppOscillation implement this.
class IHistogramPropagator {
public:
  virtual ~IHistogramPropagator() = default;

  // Returns [neutrino/antineutrino][from: nue/numu][to: nue/numu] probability
  // histograms for the given bins and oscillation parameters.
  [[nodiscard]] virtual std::array<std::array<std::array<TH2D, 2>, 2>, 2>
  GetProb_Hists(const std::vector<double> &Ebin,
                const std::vector<double> &costhbin,
                const OscillationParameters &p) = 0;

  // 3-flavour version: [nu/antinu][from: nue/numu/nutau][to: nue/numu/nutau]
  [[nodiscard]] virtual std::array<std::array<std::array<TH2D, 3>, 3>, 2>
  GetProb_Hists_3F(const std::vector<double> &Ebin,
                   const std::vector<double> &costhbin,
                   const OscillationParameters &p) = 0;
};
