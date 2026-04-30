#pragma once

#include "OscillationParameters.h"
#include "pod_hist.hpp"
#include <TH2.h>
#include <array>
#include <vector>

// Abstract interface for oscillation probability calculators that produce
// binned 2D probability histograms (E vs cos-theta).
// Both Prob3ppOscillation and ParProb3ppOscillation implement this.
class IHistogramPropagator {
public:
  virtual ~IHistogramPropagator() = default;

  virtual void re_calculate(const OscillationParameters &p) = 0;

  // ── TH2D interface (legacy, delegates to POD in Phase 7) ──────────────

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

  // ── POD interface (zero-copy–friendly) ────────────────────────────────
  // Layout: [nu/antinu][from][to] → PodHist2D<double>, same axes as TH2D version.
  // Default implementation converts TH2D → PodHist2D; override for efficiency.

  [[nodiscard]] virtual std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2>
  GetProb_Hists_POD(const std::vector<double> &Ebin,
                    const std::vector<double> &costhbin,
                    const OscillationParameters &p) {
    auto th2d = GetProb_Hists(Ebin, costhbin, p);
    std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2> ret{};
    for (int nu = 0; nu < 2; ++nu)
      for (int f = 0; f < 2; ++f)
        for (int t = 0; t < 2; ++t)
          ret[nu][f][t] = PodHist2D<double>::from_th2d(th2d[nu][f][t]);
    return ret;
  }

  [[nodiscard]] virtual std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2>
  GetProb_Hists_3F_POD(const std::vector<double> &Ebin,
                       const std::vector<double> &costhbin,
                       const OscillationParameters &p) {
    auto th2d = GetProb_Hists_3F(Ebin, costhbin, p);
    std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2> ret{};
    for (int nu = 0; nu < 2; ++nu)
      for (int f = 0; f < 3; ++f)
        for (int t = 0; t < 3; ++t)
          ret[nu][f][t] = PodHist2D<double>::from_th2d(th2d[nu][f][t]);
    return ret;
  }
};
