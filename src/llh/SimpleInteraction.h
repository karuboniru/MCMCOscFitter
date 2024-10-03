#pragma once

#include "ModelDataLLH.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataPoint.h"
#include "genie_xsec.h"
#include "hondaflux.h"
#include <functional>
#include <memory>

// extern HondaFlux flux_input;
// extern genie_xsec xsec_input;

class SimpleInteraction : public Prob3ppOscillation, public ModelDataLLH {
public:
  SimpleInteraction() : ModelDataLLH() {}
  void proposeStep() override;
  // virtual double GetLogLikelihood() const override;
  double GetLogLikelihoodAgainstData(const StateI &dataset) const override;

private:
  double weight_int{};
};

class BinnedInteraction : public Prob3ppOscillation, public ModelDataLLH {
public:
  BinnedInteraction(std::vector<double> Ebins,
                    std::vector<double> costheta_bins);
  void proposeStep() final;

  // virtual double GetLogLikelihood() const override;
  double GetLogLikelihoodAgainstData(const StateI &dataset) const final;

private:
  void UpdatePrediction();

  std::vector<double> Ebins, costheta_bins;
  std::function<TH2D(TH1D)> re_dim;
  const TH2D flux_hist_numu, flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;
  const TH2D xsec_hist_numu, xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  TH2D Prediction_hist_numu, Prediction_hist_numubar, Prediction_hist_nue,
      Prediction_hist_nuebar;
};