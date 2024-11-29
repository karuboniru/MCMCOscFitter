#pragma once

#include "ModelDataLLH.h"
#include "ParProb3ppOscillation.h"
#include "SimpleDataHist.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"
#include <functional>
#include <memory>

// extern HondaFlux flux_input;
// extern genie_xsec xsec_input;

class BinnedInteraction : public ParProb3ppOscillation, public ModelDataLLH {
public:
  BinnedInteraction(std::vector<double> Ebins,
                    std::vector<double> costheta_bins, double scale_ = 1.,
                    size_t E_rebin_factor = 1, size_t costh_rebin_factor = 1);

  BinnedInteraction(const BinnedInteraction &) = default;
  BinnedInteraction(BinnedInteraction &&) = default;
  BinnedInteraction &operator=(const BinnedInteraction &) = default;
  BinnedInteraction &operator=(BinnedInteraction &&) = default;
  ~BinnedInteraction() override = default;

  void proposeStep() final;

  // virtual double GetLogLikelihood() const override;
  double GetLogLikelihoodAgainstData(const StateI &dataset) const final;

  [[nodiscard]] SimpleDataHist GenerateData() const ;
  [[nodiscard]] SimpleDataHist GenerateData_NoOsc() const;

  void Print() const {
    flux_hist_numu.Print();
    xsec_hist_numu.Print();
  }

  void flip_hierarchy() {
    OscillationParameters::flip_hierarchy();
    re_calculate();
    UpdatePrediction();
  }

private:
  void UpdatePrediction();

  std::vector<double> Ebins, costheta_bins;
  // std::vector<double> Ebins_calc, costheta_bins_calc;
  TH2D flux_hist_numu, flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;
  TH1D xsec_hist_numu, xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  TH2D no_osc_hist_numu, no_osc_hist_numubar, no_osc_hist_nue,
      no_osc_hist_nuebar;

  TH2D Prediction_hist_numu, Prediction_hist_numubar, Prediction_hist_nue,
      Prediction_hist_nuebar;
  double scale;
  size_t E_rebin_factor;
  size_t costh_rebin_factor;
};