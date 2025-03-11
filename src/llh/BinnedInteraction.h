#pragma once

#include "ModelDataLLH.h"
#include "OscillationParameters.h"
#include "ParProb3ppOscillation.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataHist.h"
#include "genie_xsec.h"
// #include "hondaflux2d.h"
#include "WingFlux.h"
#include <format>
#include <functional>
#include <memory>

// extern HondaFlux flux_input;
// extern genie_xsec xsec_input;

using propgator_type = ParProb3ppOscillation;
// using propgator_type = Prob3ppOscillation;

class BinnedInteraction : public OscillationParameters, public ModelDataLLH {
public:
  BinnedInteraction(std::vector<double> Ebins,
                    std::vector<double> costheta_bins, double scale_ = 1.,
                    size_t E_rebin_factor = 1, size_t costh_rebin_factor = 1,
                    double IH_Bias = 1.0);

  BinnedInteraction(const BinnedInteraction &) = default;
  BinnedInteraction(BinnedInteraction &&) = default;
  BinnedInteraction &operator=(const BinnedInteraction &) = default;
  BinnedInteraction &operator=(BinnedInteraction &&) = default;
  ~BinnedInteraction() override = default;

  void proposeStep() final;

  // virtual double GetLogLikelihood() const override;
  [[nodiscard]] double
  GetLogLikelihoodAgainstData(const StateI &dataset) const final;

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

  [[nodiscard]] double GetLogLikelihood() const final;

  void UpdatePrediction();

  void SaveAs(const char *filename) const;

private:
  std::shared_ptr<propgator_type> propagator;
  std::vector<double> Ebins, costheta_bins;
  // std::vector<double> Ebins_calc, costheta_bins_calc;
  TH2D flux_hist_numu, flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;
  TH1D xsec_hist_numu, xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  // TH2D no_osc_hist_numu, no_osc_hist_numubar, no_osc_hist_nue,
  //     no_osc_hist_nuebar;

  TH2D Prediction_hist_numu, Prediction_hist_numubar, Prediction_hist_nue,
      Prediction_hist_nuebar;
  size_t E_rebin_factor;
  size_t costh_rebin_factor;

  double log_ih_bias;
};