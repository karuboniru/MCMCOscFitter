#pragma once

#include "OscillationParameters.h"
#include "pod_hist.hpp"
#include <TH2.h>
#include <array>
#include <vector>

// Abstract interface for oscillation probability calculators that produce
// binned 2D probability histograms (E vs cos-theta).
class IHistogramPropagator {
public:
  virtual ~IHistogramPropagator() = default;

  virtual void re_calculate(const OscillationParameters &p) = 0;

  // Returns [neutrino/antineutrino][from: nue/numu][to: nue/numu] probability
  // histograms as PodHist2D<double>.  Layout matches the old TH2D version:
  //   pod[nu][from][to](costh_idx, e_idx)  with costh/e indices 0-based.
  [[nodiscard]] virtual std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2>
  GetProb_Hists_POD(const std::vector<double> &Ebin,
                    const std::vector<double> &costhbin,
                    const OscillationParameters &p) = 0;

  // 3-flavour version: [nu/antinu][from: nue/numu/nutau][to: nue/numu/nutau]
  [[nodiscard]] virtual std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2>
  GetProb_Hists_3F_POD(const std::vector<double> &Ebin,
                       const std::vector<double> &costhbin,
                       const OscillationParameters &p) = 0;
};
