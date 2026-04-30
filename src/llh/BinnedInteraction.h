#pragma once

#include "IHistogramPropagator.h"
#include "OscillationParameters.h"
#include "SimpleDataHist.h"
#include "constants.h"
#include "pod_hist.hpp"
#include <TH1.h>
#include <TH2.h>
#include <format>
#include <functional>
#include <memory>
#include <vector>

// Holds the pre-computed flux and cross-section histograms needed to build a
// BinnedInteraction without calling global flux_input / xsec_input objects.
// Use this to inject known histograms in tests.
// POD members allow zero-copy construction from numpy (pybind).
struct BinnedHistograms {
  TH2D flux_numu, flux_numubar, flux_nue, flux_nuebar;
  TH1D xsec_numu, xsec_numubar, xsec_nue, xsec_nuebar;

  // POD storage — populated by pybind from numpy arrays, or lazily from TH2D.
  PodHist2D<oscillaton_calc_precision> pod_flux_numu, pod_flux_numubar, pod_flux_nue, pod_flux_nuebar;
  PodHist1D pod_xsec_numu, pod_xsec_numubar, pod_xsec_nue, pod_xsec_nuebar;
  bool pod_valid{false};
};

class BinnedInteraction : public OscillationParameters {
public:
  // Production constructor: reads flux and cross-section from the global
  // flux_input / xsec_input objects defined by the linked physics libraries.
  BinnedInteraction(std::vector<double> Ebins,
                    std::vector<double> costheta_bins, double scale_ = 1.,
                    size_t E_rebin_factor = 1, size_t costh_rebin_factor = 1,
                    double IH_Bias = 1.0);

  // Injectable constructor: accepts externally-built histograms and a
  // propagator. Useful for testing without the global physics dependencies.
  BinnedInteraction(std::vector<double> Ebins,
                    std::vector<double> costheta_bins,
                    std::shared_ptr<IHistogramPropagator> propagator,
                    BinnedHistograms histos, size_t E_rebin_factor = 1,
                    size_t costh_rebin_factor = 1, double IH_Bias = 1.0);

  BinnedInteraction(const BinnedInteraction &) = default;
  BinnedInteraction(BinnedInteraction &&) = default;
  BinnedInteraction &operator=(const BinnedInteraction &) = default;
  BinnedInteraction &operator=(BinnedInteraction &&) = default;

  void proposeStep();

  [[nodiscard]] double
  GetLogLikelihoodAgainstData(const SimpleDataHist &dataset) const;

  [[nodiscard]] SimpleDataHist GenerateData() const;
  [[nodiscard]] SimpleDataHist GenerateData_NoOsc() const;

  void Print() const {
    flux_hist_numu.Print();
    xsec_hist_numu.Print();
  }

  void flip_hierarchy() {
    OscillationParameters::flip_hierarchy();
    UpdatePrediction();
  }

  void Save_prob_hist(const std::string &name);

  [[nodiscard]] double GetLogLikelihood() const;

  void UpdatePrediction();

  void SaveAs(const char *filename) const;

private:
  void ensure_pod_flux_xsec() const;

  std::shared_ptr<IHistogramPropagator> propagator;
  std::vector<double> Ebins, costheta_bins;
  std::vector<double> Ebins_analysis, costheta_analysis;
  size_t E_rebin_factor;
  size_t costh_rebin_factor;

  size_t n_costh_fine, n_e_fine, n_costh_analysis, n_e_analysis;

  // ── Flux / xsec (TH2D/TH1D for construction, POD for computation) ────
  TH2D flux_hist_numu, flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;
  TH1D xsec_hist_numu, xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  mutable bool pod_flux_valid{false};
  mutable PodHist2D<oscillaton_calc_precision> flux_pod_numu, flux_pod_numubar, flux_pod_nue, flux_pod_nuebar;
  mutable PodHist1D xsec_pod_numu, xsec_pod_numubar, xsec_pod_nue, xsec_pod_nuebar;

  // ── Prediction (POD primary, TH2D for backward compat) ────────────────
  PodHist2D<oscillaton_calc_precision> pred_pod_numu, pred_pod_numubar, pred_pod_nue, pred_pod_nuebar;

  TH2D Prediction_hist_numu, Prediction_hist_numubar, Prediction_hist_nue,
      Prediction_hist_nuebar;

  double log_ih_bias;
};
