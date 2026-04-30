#include "Prob3ppOscillation.h"
#include "BargerPropagator.h"
#include "OscillationParameters.h"
#include <array>

[[gnu::const]]
std::array<std::array<double, 3>, 3>
Prob3ppOscillation::GetProb(int flavor, double E, double costheta,
                            const OscillationParameters &p) const {
  flavor = flavor / abs(flavor);
  BargerPropagator b(DATA_PATH "/data/density.txt");
  b.SetOneMassScaleMode(false);
  b.SetWarningSuppression(true);
  b.SetDefaultOctant(23, 2);
  b.SetMNS(p.GetT12(), p.GetT13(), p.GetT23(), p.GetDM21sq(), p.GetDM32sq(),
           p.GetDeltaCP() /*delta cp*/, E, true, flavor);
  b.DefinePath(costheta, 15);
  b.propagate(flavor);
  std::array<std::array<double, 3>, 3> ret{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ret[i][j] = b.GetProb(flavor * (i + 1), flavor * (j + 1));
    }
  }
  return ret;
}

namespace {
PodHist2D<double> barger_prob_hist_2d(const std::vector<double> &Ebin,
                                       const std::vector<double> &costhbin,
                                       int flavor, int from_idx, int to_idx,
                                       const OscillationParameters &p,
                                       const Prob3ppOscillation &prop) {
  const size_t n_e = Ebin.size() - 1;
  const size_t n_c = costhbin.size() - 1;
  PodHist2D<double> pod(n_c, n_e);
  for (size_t e_idx = 0; e_idx < n_e; ++e_idx) {
    const double emid = (Ebin[e_idx] + Ebin[e_idx + 1]) / 2.;
    for (size_t c_idx = 0; c_idx < n_c; ++c_idx) {
      const double costh = (costhbin[c_idx] + costhbin[c_idx + 1]) / 2.;
      auto prob = prop.GetProb(flavor, emid, costh, p);
      pod(c_idx, e_idx) = prob[static_cast<size_t>(from_idx)][static_cast<size_t>(to_idx)];
    }
  }
  return pod;
}
} // namespace

[[nodiscard]] std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2>
Prob3ppOscillation::GetProb_Hists_POD(const std::vector<double> &Ebin,
                                      const std::vector<double> &costhbin,
                                      const OscillationParameters &p) {
  std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2> ret{};
  for (int nu = 0; nu < 2; ++nu) {
    const int flavor = nu == 0 ? 1 : -1;
    for (int f = 0; f < 2; ++f)
      for (int t = 0; t < 2; ++t)
        ret[static_cast<size_t>(nu)][static_cast<size_t>(f)][static_cast<size_t>(t)] =
            barger_prob_hist_2d(Ebin, costhbin, flavor, f, t, p, *this);
  }
  return ret;
}

[[nodiscard]] std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2>
Prob3ppOscillation::GetProb_Hists_3F_POD(const std::vector<double> &Ebin,
                                         const std::vector<double> &costhbin,
                                         const OscillationParameters &p) {
  std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2> ret{};
  for (int nu = 0; nu < 2; ++nu) {
    const int flavor = nu == 0 ? 1 : -1;
    for (int f = 0; f < 3; ++f)
      for (int t = 0; t < 3; ++t)
        ret[static_cast<size_t>(nu)][static_cast<size_t>(f)][static_cast<size_t>(t)] =
            barger_prob_hist_2d(Ebin, costhbin, flavor, f, t, p, *this);
  }
  return ret;
}

void Prob3ppOscillation::re_calculate(const OscillationParameters &) {
  // BargerPropagator computes probabilities on-demand in GetProb_Hists_POD.
  // No pre-calculation needed for the CPU path.
}
