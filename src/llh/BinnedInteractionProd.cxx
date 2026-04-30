// Production constructor for BinnedInteraction.
// Separated from BinnedInteraction.cxx so that the injectable constructor and
// all other methods can be compiled without pulling in the HondaFlux2D and
// GENIE_XSEC global objects (which require physics data files at load time).

#include "BinnedInteraction.h"
#include "ParProb3ppOscillation.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"

namespace {
template <typename T>
concept is_hist = std::is_base_of_v<TH1, T>;

template <is_hist T> T operator*(const T &lhs, double rhs) {
  T ret = lhs;
  ret.Scale(rhs);
  return ret;
}
} // namespace

BinnedInteraction::BinnedInteraction(std::vector<double> Ebins_,
                                     std::vector<double> costheta_bins_,
                                     double scale_, size_t E_rebin_factor_,
                                     size_t costh_rebin_factor_,
                                     double IH_bias_)
    : propagator{std::make_shared<ParProb3ppOscillation>(
          to_center<oscillaton_calc_precision>(Ebins_),
          to_center<oscillaton_calc_precision>(costheta_bins_))},
      Ebins(std::move(Ebins_)), costheta_bins(std::move(costheta_bins_)),
      E_rebin_factor(E_rebin_factor_), costh_rebin_factor(costh_rebin_factor_),
      n_costh_fine(costheta_bins.size() - 1),
      n_e_fine(Ebins.size() - 1),
      n_costh_analysis(n_costh_fine / costh_rebin_factor_),
      n_e_analysis(n_e_fine / E_rebin_factor_),
      flux_hist_numu(flux_input.GetFlux_Hist(Ebins, costheta_bins, 14) *
                     scale_),
      flux_hist_numubar(flux_input.GetFlux_Hist(Ebins, costheta_bins, -14) *
                        scale_),
      flux_hist_nue(flux_input.GetFlux_Hist(Ebins, costheta_bins, 12) * scale_),
      flux_hist_nuebar(flux_input.GetFlux_Hist(Ebins, costheta_bins, -12) *
                       scale_),
      xsec_hist_numu(xsec_input.GetXsecHistMixture(
          Ebins, 14, {{1000060120, 1.0}, {2212, H_to_C}})),
      xsec_hist_numubar(xsec_input.GetXsecHistMixture(
          Ebins, -14, {{1000060120, 1.0}, {2212, H_to_C}})),
      xsec_hist_nue(xsec_input.GetXsecHistMixture(
          Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}})),
      xsec_hist_nuebar(xsec_input.GetXsecHistMixture(
          Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}})),
      log_ih_bias(std::log(IH_bias_)) {
  Ebins_analysis.reserve(n_e_analysis + 1);
  for (size_t i = 0; i <= n_e_analysis * E_rebin_factor; i += E_rebin_factor)
    Ebins_analysis.push_back(Ebins[i]);
  costheta_analysis.reserve(n_costh_analysis + 1);
  for (size_t i = 0; i <= n_costh_analysis * costh_rebin_factor; i += costh_rebin_factor)
    costheta_analysis.push_back(costheta_bins[i]);
  UpdatePrediction();
}
