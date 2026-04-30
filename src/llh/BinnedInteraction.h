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
struct BinnedHistograms {
  PodHist2D<oscillaton_calc_precision> pod_flux_numu, pod_flux_numubar, pod_flux_nue, pod_flux_nuebar;
  PodHist1D pod_xsec_numu, pod_xsec_numubar, pod_xsec_nue, pod_xsec_nuebar;
};

// Immutable data shared between BinnedInteraction copies (eliminates
// deep-copying flux/xsec/bin-edges during MCMC).  Constructed once,
// referenced via shared_ptr with refcount bump on copy.
struct BinnedInteractionImmutable {
  std::vector<double> Ebins, costheta_bins;
  std::vector<double> Ebins_analysis, costheta_analysis;
  size_t E_rebin_factor{}, costh_rebin_factor{};
  size_t n_costh_fine{}, n_e_fine{}, n_costh_analysis{}, n_e_analysis{};
  PodHist2D<oscillaton_calc_precision> flux_numu, flux_numubar, flux_nue, flux_nuebar;
  PodHist1D xsec_numu, xsec_numubar, xsec_nue, xsec_nuebar;
  double log_ih_bias{};
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

  BinnedInteraction(const BinnedInteraction &other)
      : OscillationParameters(other), propagator(other.propagator),
        imm_(other.imm_),
        pred_pod_numu(other.pred_pod_numu),
        pred_pod_numubar(other.pred_pod_numubar),
        pred_pod_nue(other.pred_pod_nue),
        pred_pod_nuebar(other.pred_pod_nuebar) {}
  BinnedInteraction(BinnedInteraction &&) = default;
  BinnedInteraction &operator=(const BinnedInteraction &other) {
    if (this != &other) {
      OscillationParameters::operator=(other);
      propagator = other.propagator;
      imm_ = other.imm_;
      pred_pod_numu    = other.pred_pod_numu;
      pred_pod_numubar = other.pred_pod_numubar;
      pred_pod_nue     = other.pred_pod_nue;
      pred_pod_nuebar  = other.pred_pod_nuebar;
    }
    return *this;
  }
  BinnedInteraction &operator=(BinnedInteraction &&) = default;

  void proposeStep();

  [[nodiscard]] double
  GetLogLikelihoodAgainstData(const SimpleDataHist &dataset) const;

  [[nodiscard]] SimpleDataHist GenerateData() const;
  [[nodiscard]] SimpleDataHist GenerateData_NoOsc() const;

  void Print() const {}

  void flip_hierarchy() {
    OscillationParameters::flip_hierarchy();
    UpdatePrediction();
  }

  void Save_prob_hist(const std::string &name);

  [[nodiscard]] double GetLogLikelihood() const;

  void UpdatePrediction();

  void SaveAs(const char *filename) const;

private:
  std::shared_ptr<IHistogramPropagator> propagator;
  std::shared_ptr<const BinnedInteractionImmutable> imm_;

  // ── Prediction (mutable — deep-copied on copy) ──────────────────────
  PodHist2D<double> pred_pod_numu, pred_pod_numubar, pred_pod_nue, pred_pod_nuebar;
};
