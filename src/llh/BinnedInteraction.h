#pragma once

#include "IHistogramPropagator.h"
#include "OscillationParameters.h"
#include "SimpleDataHist.h"
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
  TH2D flux_numu, flux_numubar, flux_nue, flux_nuebar;
  TH1D xsec_numu, xsec_numubar, xsec_nue, xsec_nuebar;
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
  std::shared_ptr<IHistogramPropagator> propagator;
  std::vector<double> Ebins, costheta_bins;
  TH2D flux_hist_numu, flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;
  TH1D xsec_hist_numu, xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  TH2D Prediction_hist_numu, Prediction_hist_numubar, Prediction_hist_nue,
      Prediction_hist_nuebar;
  size_t E_rebin_factor;
  size_t costh_rebin_factor;

  double log_ih_bias;
};
