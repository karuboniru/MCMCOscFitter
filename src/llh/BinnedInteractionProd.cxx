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
#include "pod_hist.hpp"

BinnedInteraction::BinnedInteraction(std::vector<double> Ebins_,
                                     std::vector<double> costheta_bins_,
                                     double scale_, size_t E_rebin_factor_,
                                     size_t costh_rebin_factor_,
                                     double IH_bias_)
    : propagator{std::make_shared<ParProb3ppOscillation>(
          to_center<oscillaton_calc_precision>(Ebins_),
          to_center<oscillaton_calc_precision>(costheta_bins_))} {
  auto imm = std::make_shared<BinnedInteractionImmutable>();
  imm->Ebins = std::move(Ebins_);
  imm->costheta_bins = std::move(costheta_bins_);
  imm->E_rebin_factor = E_rebin_factor_;
  imm->costh_rebin_factor = costh_rebin_factor_;
  imm->n_costh_fine = imm->costheta_bins.size() - 1;
  imm->n_e_fine = imm->Ebins.size() - 1;
  imm->n_costh_analysis = imm->n_costh_fine / costh_rebin_factor_;
  imm->n_e_analysis = imm->n_e_fine / E_rebin_factor_;

  imm->flux_numu    = PodHist2D<oscillaton_calc_precision>::from_th2d(
      flux_input.GetFlux_Hist(imm->Ebins, imm->costheta_bins, 14) * scale_);
  imm->flux_numubar = PodHist2D<oscillaton_calc_precision>::from_th2d(
      flux_input.GetFlux_Hist(imm->Ebins, imm->costheta_bins, -14) * scale_);
  imm->flux_nue     = PodHist2D<oscillaton_calc_precision>::from_th2d(
      flux_input.GetFlux_Hist(imm->Ebins, imm->costheta_bins, 12) * scale_);
  imm->flux_nuebar  = PodHist2D<oscillaton_calc_precision>::from_th2d(
      flux_input.GetFlux_Hist(imm->Ebins, imm->costheta_bins, -12) * scale_);

  imm->xsec_numu    = th1d_to_pod(xsec_input.GetXsecHistMixture(
      imm->Ebins, 14, {{1000060120, 1.0}, {2212, H_to_C}}));
  imm->xsec_numubar = th1d_to_pod(xsec_input.GetXsecHistMixture(
      imm->Ebins, -14, {{1000060120, 1.0}, {2212, H_to_C}}));
  imm->xsec_nue     = th1d_to_pod(xsec_input.GetXsecHistMixture(
      imm->Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}}));
  imm->xsec_nuebar  = th1d_to_pod(xsec_input.GetXsecHistMixture(
      imm->Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}}));

  imm->log_ih_bias  = std::log(IH_bias_);

  imm->Ebins_analysis.reserve(imm->n_e_analysis + 1);
  for (size_t i = 0; i <= imm->n_e_analysis * imm->E_rebin_factor; i += imm->E_rebin_factor)
    imm->Ebins_analysis.push_back(imm->Ebins[i]);
  imm->costheta_analysis.reserve(imm->n_costh_analysis + 1);
  for (size_t i = 0; i <= imm->n_costh_analysis * imm->costh_rebin_factor; i += imm->costh_rebin_factor)
    imm->costheta_analysis.push_back(imm->costheta_bins[i]);

  imm_ = std::move(imm);
  UpdatePrediction();
}
