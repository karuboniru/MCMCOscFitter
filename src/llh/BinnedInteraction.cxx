
#include "BinnedInteraction.h"
#include "OscillationParameters.h"
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
                                     size_t costh_rebin_factor_,
                                     double IH_bias_)
    : ModelDataLLH(),
      propagator{std::make_shared<propgator_type>(
          to_center<oscillaton_calc_precision>(Ebins_),
          to_center<oscillaton_calc_precision>(costheta_bins_))},
      Ebins(std::move(Ebins_)), costheta_bins(std::move(costheta_bins_)),
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
      E_rebin_factor(E_rebin_factor_), costh_rebin_factor(costh_rebin_factor_),
      log_ih_bias(std::log(IH_bias_)) {
  UpdatePrediction();
}

void BinnedInteraction::UpdatePrediction() {
  // [0-neutrino, 1-antineutrino][from: 0-nue, 1-mu][to: 0-e, 1-mu]
  auto oscHists = propagator->GetProb_Hists(Ebins, costheta_bins, *this);

  // auto oscHist = GetProb_Hist(Ebins, costheta_bins, 1);
  // auto oscHist_anti = GetProb_Hist(Ebins, costheta_bins, -1);
  auto &oscHist = oscHists[0];
  auto &oscHist_anti = oscHists[1];

  auto osc_flux_numu =
      oscHist[0][1] * flux_hist_nue + oscHist[1][1] * flux_hist_numu;
  auto osc_flux_nue =
      oscHist[0][0] * flux_hist_nue + oscHist[1][0] * flux_hist_numu;

  auto osc_flux_numubar = oscHist_anti[0][1] * flux_hist_nuebar +
                          oscHist_anti[1][1] * flux_hist_numubar;
  auto osc_flux_nuebar = oscHist_anti[0][0] * flux_hist_nuebar +
                         oscHist_anti[1][0] * flux_hist_numubar;

  Prediction_hist_numu = osc_flux_numu * xsec_hist_numu;
  Prediction_hist_numubar = osc_flux_numubar * xsec_hist_numubar;
  Prediction_hist_nue = osc_flux_nue * xsec_hist_nue;
  Prediction_hist_nuebar = osc_flux_nuebar * xsec_hist_nuebar;

  Prediction_hist_numu.Rebin2D(E_rebin_factor, costh_rebin_factor);
  Prediction_hist_numubar.Rebin2D(E_rebin_factor, costh_rebin_factor);
  Prediction_hist_nue.Rebin2D(E_rebin_factor, costh_rebin_factor);
  Prediction_hist_nuebar.Rebin2D(E_rebin_factor, costh_rebin_factor);
}

void BinnedInteraction::proposeStep() {
  OscillationParameters::proposeStep();
  UpdatePrediction();
}

namespace {
double TH2D_chi2(const TH2D &data, const TH2D &pred) {
  auto binsx = data.GetNbinsX();
  auto binsy = data.GetNbinsY();
  double chi2{};
  // #pragma omp parallel for reduction(+ : chi2) collapse(2)
  for (int x = 1; x <= binsx; x++) {
    for (int y = 1; y <= binsy; y++) {
      auto bin_data = data.GetBinContent(x, y);
      auto bin_pred = pred.GetBinContent(x, y);
      if (bin_data != 0) [[likely]]
        chi2 +=
            (bin_pred - bin_data) + bin_data * std::log(bin_data / bin_pred);
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
  auto chi2_nue = TH2D_chi2(data_hist_nue, Prediction_hist_nue);
  auto chi2_nuebar = TH2D_chi2(data_hist_nuebar, Prediction_hist_nuebar);

  auto llh = -0.5 * (chi2_numu + chi2_numubar + chi2_nue + chi2_nuebar);
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

double BinnedInteraction::GetLogLikelihood() const {
  auto llh = OscillationParameters::GetLogLikelihood();
  if (GetDM32sq() < 0) {
    llh += log_ih_bias;
  }
  return llh;
}

void BinnedInteraction::SaveAs(const char *filename) const {
  auto file = TFile::Open(filename, "RECREATE");
  file->cd();

  file->Add(flux_hist_numu.Clone("flux_hist_numu"));
  file->Add(flux_hist_numubar.Clone("flux_hist_numubar"));
  file->Add(flux_hist_nue.Clone("flux_hist_nue"));
  file->Add(flux_hist_nuebar.Clone("flux_hist_nuebar"));

  file->Add(xsec_hist_numu.Clone("xsec_hist_numu"));
  file->Add(xsec_hist_numubar.Clone("xsec_hist_numubar"));
  file->Add(xsec_hist_nue.Clone("xsec_hist_nue"));
  file->Add(xsec_hist_nuebar.Clone("xsec_hist_nuebar"));

  file->Add(Prediction_hist_numu.Clone("Prediction_hist_numu"));
  file->Add(Prediction_hist_numubar.Clone("Prediction_hist_numubar"));
  file->Add(Prediction_hist_nue.Clone("Prediction_hist_nue"));
  file->Add(Prediction_hist_nuebar.Clone("Prediction_hist_nuebar"));

  file->Write();
  file->Close();
  delete file;
}

void BinnedInteraction::Save_prob_hist(const std::string &name) {
  // if constexpr (std::is_same_v<ParProb3ppOscillation, propgator_type>) {
  auto file = TFile::Open(name.c_str(), "RECREATE");
  file->cd();
  auto prob_hist = propagator->GetProb_Hists_3F(Ebins, costheta_bins, *this);
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
  file->Close();
  delete file;
}