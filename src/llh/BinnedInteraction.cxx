// All BinnedInteraction methods except the production constructor.
// The production constructor lives in BinnedInteractionProd.cxx and depends on
// the HondaFlux2D and GENIE_XSEC global objects.  Keeping them separate lets
// pybind (and tests) link this TU without the physics-data-file globals.

#include "BinnedInteraction.h"
#include "OscillationParameters.h"
#include "chi2.h"

#include <SimpleDataHist.h>
#include <TFile.h>
#include <TH2.h>
#include <format>
#include <type_traits>
#include <utility>

namespace {
template <typename T>
concept is_hist = std::is_base_of_v<TH1, T>;

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

// Injectable constructor: all histograms and propagator supplied externally.
BinnedInteraction::BinnedInteraction(std::vector<double> Ebins_,
                                     std::vector<double> costheta_bins_,
                                     std::shared_ptr<IHistogramPropagator> prop,
                                     BinnedHistograms histos,
                                     size_t E_rebin_factor_,
                                     size_t costh_rebin_factor_,
                                     double IH_bias_)
    : propagator{std::move(prop)},
      Ebins(std::move(Ebins_)), costheta_bins(std::move(costheta_bins_)),
      flux_hist_numu(std::move(histos.flux_numu)),
      flux_hist_numubar(std::move(histos.flux_numubar)),
      flux_hist_nue(std::move(histos.flux_nue)),
      flux_hist_nuebar(std::move(histos.flux_nuebar)),
      xsec_hist_numu(std::move(histos.xsec_numu)),
      xsec_hist_numubar(std::move(histos.xsec_numubar)),
      xsec_hist_nue(std::move(histos.xsec_nue)),
      xsec_hist_nuebar(std::move(histos.xsec_nuebar)),
      E_rebin_factor(E_rebin_factor_), costh_rebin_factor(costh_rebin_factor_),
      log_ih_bias(std::log(IH_bias_)) {
  UpdatePrediction();
}

void BinnedInteraction::UpdatePrediction() {
  // [0-neutrino, 1-antineutrino][from: 0-nue, 1-numu][to: 0-nue, 1-numu]
  auto oscHists = propagator->GetProb_Hists(Ebins, costheta_bins, *this);

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

double
BinnedInteraction::GetLogLikelihoodAgainstData(const SimpleDataHist &dataset) const {
  auto chi2_numu    = TH2D_chi2(dataset.hist_numu,    Prediction_hist_numu);
  auto chi2_numubar = TH2D_chi2(dataset.hist_numubar, Prediction_hist_numubar);
  auto chi2_nue     = TH2D_chi2(dataset.hist_nue,     Prediction_hist_nue);
  auto chi2_nuebar  = TH2D_chi2(dataset.hist_nuebar,  Prediction_hist_nuebar);
  return -0.5 * (chi2_numu + chi2_numubar + chi2_nue + chi2_nuebar);
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
  if (GetDM32sq() < 0)
    llh += log_ih_bias;
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
  auto file = TFile::Open(name.c_str(), "RECREATE");
  file->cd();
  auto prob_hist = propagator->GetProb_Hists_3F(Ebins, costheta_bins, *this);
  auto id_2_name = std::to_array({"nue", "numu", "nutau"});
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        prob_hist[i][j][k].SetName(
            std::format("{}_{}_{}", i == 0 ? "neutrino" : "antineutrino",
                        id_2_name[j], id_2_name[k])
                .c_str());
        prob_hist[i][j][k].Write();
      }
  file->Close();
  delete file;
}
