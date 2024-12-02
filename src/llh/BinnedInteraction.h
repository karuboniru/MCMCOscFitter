#pragma once

#include "ModelDataLLH.h"
#include "ParProb3ppOscillation.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataHist.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"
#include <format>
#include <functional>
#include <memory>

// extern HondaFlux flux_input;
// extern genie_xsec xsec_input;

using propgator_type = ParProb3ppOscillation;
// using propgator_type = Prob3ppOscillation;

class BinnedInteraction : public propgator_type, public ModelDataLLH {
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

  [[nodiscard]] SimpleDataHist GenerateData() const;
  [[nodiscard]] SimpleDataHist GenerateData_NoOsc() const;

  void Print() const {
    flux_hist_numu.Print();
    xsec_hist_numu.Print();
  }

  void flip_hierarchy() {
    OscillationParameters::flip_hierarchy();
    if constexpr (std::is_same_v<ParProb3ppOscillation, propgator_type>) {
      re_calculate();
    }
    UpdatePrediction();
  }

  void Save_prob_hist(const std::string &name) {
    // if constexpr (std::is_same_v<ParProb3ppOscillation, propgator_type>) {
    auto file = TFile::Open(name.c_str(), "RECREATE");
    file->cd();
    auto prob_hist = GetProb_Hists_3F(Ebins, costheta_bins);
    auto id_2_name = std::to_array({"nue", "numu", "nutau"});
    // prob_hist: [0 neutrino, 1 antineutrino][from: 0-nue, 1-mu][to: 0-e,
    // 1-mu]
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          prob_hist[i][j][k].SetName(
              std::format("{}_{}_{}", i == 0 ? "neutrino" : "antineutrino",
                          id_2_name[j], id_2_name[k])
                  .c_str());
          prob_hist[i][j][k].Write();
        }
      }
    }
    // file->Write();
    file->Close();
    delete file;
    // }
  }

private:
  void UpdatePrediction();

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
};