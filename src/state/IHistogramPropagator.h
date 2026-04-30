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

  // ── Raw probability access (optional, for bulk propagators) ────────────
  // Propagators that pre-compute probability arrays on re_calculate() may
  // expose them via these methods, eliminating the ~6 MB PodHist2D<double>
  // intermediate allocation in UpdatePrediction.  Layout:
  //   [ProbType * n_cosines * n_energies + costh_idx * n_energies + e_idx]
  // ProbType: e_e=0, e_m=1, e_t=2, m_e=3, m_m=4, m_t=5, t_e=6, t_m=7, t_t=8.
  [[nodiscard]] virtual bool has_raw_results() const { return false; }
  [[nodiscard]] virtual const oscillaton_calc_precision *raw_prob_neutrino() const { return nullptr; }
  [[nodiscard]] virtual const oscillaton_calc_precision *raw_prob_antineutrino() const { return nullptr; }
  [[nodiscard]] virtual size_t raw_n_cosines() const { return 0; }
  [[nodiscard]] virtual size_t raw_n_energies() const { return 0; }
};
