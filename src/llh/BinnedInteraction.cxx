
#include "BinnedInteraction.h"
#include "ParProb3ppOscillation.h"
#include "binning_tool.hpp"
#include "genie_xsec.h"
#include "hondaflux2d.h"

#include <SimpleDataHist.h>
#include <TF3.h>
#include <TH2.h>
#include <type_traits>
#include <utility>

BinnedInteraction::BinnedInteraction(std::vector<double> Ebins_,
                                     std::vector<double> costheta_bins_,
                                     double scale_, size_t E_rebin_factor_)
    : ParProb3ppOscillation{to_center<float>(Ebins_),
                            linspace<float>(
                                costheta_bins_[0],
                                costheta_bins_[costheta_bins_.size() - 1],
                                costheta_bins_.size() * 50)},
      ModelDataLLH(), Ebins(std::move(Ebins_)),
      costheta_bins(std::move(costheta_bins_)),
      // Ebins_calc(logspace<double>(Ebins[0], Ebins[Ebins.size() - 1],
      //                             Ebins.size() * 10)),
      // costheta_bins_calc(linspace<double>(
      //     costheta_bins[0], costheta_bins[costheta_bins.size() - 1],
      //     costheta_bins.size() * 10)),
      re_dim([&](const TH1D &hist) -> TH2D {
        TH2D re_hist("", "", Ebins.size() - 1, Ebins.data(),
                     costheta_bins.size() - 1, costheta_bins.data());
        for (int i = 0; i < re_hist.GetNbinsX(); i++) {
          for (int j = 0; j < re_hist.GetNbinsY(); j++) {
            re_hist.SetBinContent(i + 1, j + 1, hist.GetBinContent(i + 1));
          }
        }
        return re_hist;
      }),
      flux_hist_numu(flux_input.GetFlux_Hist(Ebins, costheta_bins, 14)),
      flux_hist_numubar(flux_input.GetFlux_Hist(Ebins, costheta_bins, -14)),
      flux_hist_nue(flux_input.GetFlux_Hist(Ebins, costheta_bins, 12)),
      flux_hist_nuebar(flux_input.GetFlux_Hist(Ebins, costheta_bins, -12)),
      xsec_hist_numu(re_dim(xsec_input.GetXsecHist(Ebins, 14, 1000060120))),
      xsec_hist_numubar(re_dim(xsec_input.GetXsecHist(Ebins, -14, 1000060120))),
      xsec_hist_nue(re_dim(xsec_input.GetXsecHist(Ebins, 12, 1000060120))),
      xsec_hist_nuebar(re_dim(xsec_input.GetXsecHist(Ebins, -12, 1000060120))),
      scale(scale_), E_rebin_factor(E_rebin_factor_) {
  UpdatePrediction();
}

// just to avoid any potential conflict
namespace {
template <typename T>
concept is_hist = std::is_base_of_v<TH1, T>;

template <is_hist T> T operator*(const T &lhs, const T &rhs) {
  T ret = lhs;
  ret.Multiply(&rhs);
  return ret;
}

template <is_hist T> T operator+(const T &lhs, const T &rhs) {
  T ret = lhs;
  ret.Add(&rhs);
  return ret;
}

template <is_hist T> T operator*(const T &lhs, double rhs) {
  T ret = lhs;
  ret.Scale(rhs);
  return ret;
}

template <is_hist T> T operator*(double lhs, const T &rhs) { return rhs * lhs; }

} // namespace

void BinnedInteraction::UpdatePrediction() {
  // auto oscHist = GetProb_Hist(Ebins, costheta_bins, 12);
  // auto oscHist_anti = GetProb_Hist(Ebins, costheta_bins, -12);
  auto oscHists = GetProb_Hist(Ebins, costheta_bins);
  auto &oscHist = oscHists[0];
  auto &oscHist_anti = oscHists[1];
  Prediction_hist_numu =
      scale * (oscHist[1][1] * flux_hist_numu * xsec_hist_numu +
               oscHist[0][1] * flux_hist_nue * xsec_hist_nue);
  Prediction_hist_numubar =
      scale * (oscHist_anti[1][0] * flux_hist_numubar * xsec_hist_numubar +
               oscHist_anti[0][0] * flux_hist_nuebar * xsec_hist_nuebar);
  Prediction_hist_nue =
      scale * (oscHist[0][1] * flux_hist_nue * xsec_hist_nue +
               oscHist[1][1] * flux_hist_numu * xsec_hist_numu);
  Prediction_hist_nuebar =
      scale * (oscHist_anti[0][0] * flux_hist_nuebar * xsec_hist_nuebar +
               oscHist_anti[1][0] * flux_hist_numubar * xsec_hist_numubar);
  Prediction_hist_numu.Rebin2D(E_rebin_factor, 1);
  Prediction_hist_numubar.Rebin2D(E_rebin_factor, 1);
  Prediction_hist_nue.Rebin2D(E_rebin_factor, 1);
  Prediction_hist_nuebar.Rebin2D(E_rebin_factor, 1);
}

void BinnedInteraction::proposeStep() {
  ParProb3ppOscillation::proposeStep();
  UpdatePrediction();
}

namespace {
double TH2D_chi2(const TH2D &data, const TH2D &pred) {
  auto binsx = data.GetNbinsX();
  auto binsy = data.GetNbinsY();
  double chi2{};
  for (int x = 1; x <= binsx; x++) {
    for (int y = 1; y <= binsy; y++) {
      auto bin_pred = pred.GetBinContent(x, y);
      if (bin_pred < 5.) {
        continue;
      }
      auto bin_data = data.GetBinContent(x, y);
      // auto diff = bin_data - bin_pred;
      // chi2 += diff * diff / bin_pred;
      chi2 += (bin_pred - bin_data) + bin_data * log(bin_data / bin_pred);
    }
  }
  return 2 * chi2;
}
} // namespace
double
BinnedInteraction::GetLogLikelihoodAgainstData(StateI const &dataset) const {
  auto casted = dynamic_cast<const SimpleDataHist &>(dataset);
  auto &data_hist_numu = casted.hist_numu;
  auto &data_hist_numubar = casted.hist_numubar;
  auto &data_hist_nue = casted.hist_nue;
  auto &data_hist_nuebar = casted.hist_nuebar;

  // auto chi2_numu = Prediction_hist_numu.Chi2Test(&data_hist_numu, "CHI2");
  // auto chi2_numubar =
  //     Prediction_hist_numubar.Chi2Test(&data_hist_numubar, "CHI2");
  // auto combined_prediction = Prediction_hist_numu + Prediction_hist_numubar;
  // auto combined_data = data_hist_numu + data_hist_numubar;
  // auto combined_prediction_e = Prediction_hist_nue + Prediction_hist_nuebar;
  // auto combined_data_e = data_hist_nue + data_hist_nuebar;
  // auto chi2_numu = combined_data.Chi2Test(&combined_prediction, "CHI2 UW");
  // auto chi2_nue = Prediction_hist_nue.Chi2Test(&data_hist_nue, "CHI2");
  // auto chi2_nuebar = Prediction_hist_nuebar.Chi2Test(&data_hist_nuebar,
  // "CHI2");
  auto chi2_numu = TH2D_chi2(data_hist_numu, Prediction_hist_numu);
  auto chi2_numubar = TH2D_chi2(data_hist_numubar, Prediction_hist_numubar);
  auto chi2_nue = TH2D_chi2(data_hist_nue + data_hist_nuebar,
                            Prediction_hist_nue + Prediction_hist_nuebar);
  // auto chi2_nuebar = TH2D_chi2(data_hist_nuebar, Prediction_hist_nuebar);

  // auto chi2_nue = TH2D_chi2(combined_data_e, combined_prediction_e);
  auto llh = -0.5 * (chi2_numu + chi2_numubar + chi2_nue);
  // llh *= 5;
  return llh;
}

SimpleDataHist BinnedInteraction::GenerateData() {
  SimpleDataHist data;
  data.hist_numu = Prediction_hist_numu;
  data.hist_numubar = Prediction_hist_numubar;
  data.hist_nue = Prediction_hist_nue;
  data.hist_nuebar = Prediction_hist_nuebar;
  return data;
}
