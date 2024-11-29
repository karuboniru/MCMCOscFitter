
#include "BinnedInteraction.h"
#include "ParProb3ppOscillation.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"

#include <SimpleDataHist.h>
#include <TF3.h>
#include <TH2.h>
#include <type_traits>
#include <utility>

// just to avoid any potential conflict
namespace {
template <typename T>
concept is_hist = std::is_base_of_v<TH1, T>;

template <typename T>
concept is_hist2D = std::is_base_of_v<TH2, T>;

template <is_hist T> T operator*(const T &lhs, const T &rhs) {
  T ret = lhs;
  ret.Multiply(&rhs);
  return ret;
}

TH2D operator*(const TH2D &lhs, const TH1D &rhs) {
  TH2D ret = lhs;

  assert(lhs.GetNbinsX() == rhs.GetNbinsX());

  for (int x_index = 1; x_index <= lhs.GetNbinsX(); x_index++) {
    for (int y_index = 1; y_index <= lhs.GetNbinsY(); y_index++) {
      ret.SetBinContent(x_index, y_index,
                        ret.GetBinContent(x_index, y_index) *
                            rhs.GetBinContent(x_index));
    }
  }
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

template <is_hist T> T operator*(double lhs, const T &&rhs) {
  return rhs * lhs;
}

} // namespace

BinnedInteraction::BinnedInteraction(std::vector<double> Ebins_,
                                     std::vector<double> costheta_bins_,
                                     double scale_, size_t E_rebin_factor_,
                                     size_t costh_rebin_factor_)
    : ParProb3ppOscillation{to_center<float>(Ebins_),
                            to_center<float>(costheta_bins_)},
      ModelDataLLH(), Ebins(std::move(Ebins_)),
      costheta_bins(std::move(costheta_bins_)),
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
      scale(scale_), E_rebin_factor(E_rebin_factor_),
      costh_rebin_factor(costh_rebin_factor_) {
  UpdatePrediction();
}

void BinnedInteraction::UpdatePrediction() {
  // [0-neutrino, 1-antineutrino][from: 0-nue, 1-mu][to: 0-e, 1-mu]
  auto oscHists = GetProb_Hist(Ebins, costheta_bins);
  auto &oscHist = oscHists[0];
  auto &oscHist_anti = oscHists[1];
  // auto no_osc_e = flux_hist_nue * xsec_hist_nue;
  // auto no_osc_ebar = flux_hist_nuebar * xsec_hist_nuebar;
  // auto no_osc_mu = flux_hist_numu * xsec_hist_numu;
  // auto no_osc_mubar = flux_hist_numubar * xsec_hist_numubar;
  auto no_osc_e = flux_hist_nue * xsec_hist_nue;
  auto no_osc_ebar = flux_hist_nuebar * xsec_hist_nuebar;
  auto no_osc_mu = flux_hist_numu * xsec_hist_numu;
  auto no_osc_mubar = flux_hist_numubar * xsec_hist_numubar;
  Prediction_hist_numu = (oscHist[1][1] * no_osc_mu + oscHist[0][1] * no_osc_e);
  Prediction_hist_numubar =
      (oscHist_anti[1][1] * no_osc_mubar + oscHist_anti[0][1] * no_osc_ebar);
  Prediction_hist_nue = (oscHist[0][0] * no_osc_e + oscHist[1][0] * no_osc_mu);
  Prediction_hist_nuebar =
      (oscHist_anti[0][0] * no_osc_ebar + oscHist_anti[1][0] * no_osc_mubar);

  Prediction_hist_numu.Rebin2D(E_rebin_factor, costh_rebin_factor);
  Prediction_hist_numubar.Rebin2D(E_rebin_factor, costh_rebin_factor);
  Prediction_hist_nue.Rebin2D(E_rebin_factor, costh_rebin_factor);
  Prediction_hist_nuebar.Rebin2D(E_rebin_factor, costh_rebin_factor);
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
      auto bin_data = data.GetBinContent(x, y);
      auto bin_pred = pred.GetBinContent(x, y);
      if (bin_pred < 3.) {
        continue;
      }
      // auto diff = bin_data - bin_pred;
      // chi2 += diff * diff / bin_pred;
      if (bin_data != 0) [[likely]]
        chi2 += (bin_pred - bin_data) + bin_data * log(bin_data / bin_pred);
      else
        chi2 += bin_pred;
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

  auto chi2_numu = TH2D_chi2(data_hist_numu, Prediction_hist_numu);
  auto chi2_numubar = TH2D_chi2(data_hist_numubar, Prediction_hist_numubar);
  auto chi2_nue = TH2D_chi2(data_hist_nue + data_hist_nuebar,
                            Prediction_hist_nue + Prediction_hist_nuebar);
  auto chi2_nuebar = TH2D_chi2(data_hist_nuebar, Prediction_hist_nuebar);

  auto llh = -0.5 * (chi2_numu + chi2_numubar + chi2_nue + chi2_nuebar);
  // llh *= 5;
  return llh;
}

SimpleDataHist BinnedInteraction::GenerateData() const {
  SimpleDataHist data;
  data.hist_numu = Prediction_hist_numu;
  data.hist_numubar = Prediction_hist_numubar;
  data.hist_nue = Prediction_hist_nue;
  data.hist_nuebar = Prediction_hist_nuebar;
  return data;
}

SimpleDataHist BinnedInteraction::GenerateData_NoOsc() const {
  SimpleDataHist data;
  data.hist_numu = flux_hist_numu * xsec_hist_numu;
  data.hist_numubar = flux_hist_numubar * xsec_hist_numubar;
  data.hist_nue = flux_hist_nue * xsec_hist_nue;
  data.hist_nuebar = flux_hist_nuebar * xsec_hist_nuebar;
  return data;
}
