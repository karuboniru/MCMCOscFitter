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
      E_rebin_factor(E_rebin_factor_), costh_rebin_factor(costh_rebin_factor_),
      n_costh_fine(costheta_bins.size() - 1),
      n_e_fine(Ebins.size() - 1),
      n_costh_analysis(n_costh_fine / costh_rebin_factor_),
      n_e_analysis(n_e_fine / E_rebin_factor_),
      flux_hist_numu(std::move(histos.flux_numu)),
      flux_hist_numubar(std::move(histos.flux_numubar)),
      flux_hist_nue(std::move(histos.flux_nue)),
      flux_hist_nuebar(std::move(histos.flux_nuebar)),
      xsec_hist_numu(std::move(histos.xsec_numu)),
      xsec_hist_numubar(std::move(histos.xsec_numubar)),
      xsec_hist_nue(std::move(histos.xsec_nue)),
      xsec_hist_nuebar(std::move(histos.xsec_nuebar)),
      // Take POD data directly if the caller supplied it (avoids TH2D→POD conversion).
      flux_pod_numu(std::move(histos.pod_flux_numu)),
      flux_pod_numubar(std::move(histos.pod_flux_numubar)),
      flux_pod_nue(std::move(histos.pod_flux_nue)),
      flux_pod_nuebar(std::move(histos.pod_flux_nuebar)),
      xsec_pod_numu(std::move(histos.pod_xsec_numu)),
      xsec_pod_numubar(std::move(histos.pod_xsec_numubar)),
      xsec_pod_nue(std::move(histos.pod_xsec_nue)),
      xsec_pod_nuebar(std::move(histos.pod_xsec_nuebar)),
      pod_flux_valid(histos.pod_valid),
      log_ih_bias(std::log(IH_bias_)) {
  // Compute analysis bin edges by striding fine bin edges.
  Ebins_analysis.reserve(n_e_analysis + 1);
  for (size_t i = 0; i <= n_e_analysis * E_rebin_factor; i += E_rebin_factor)
    Ebins_analysis.push_back(Ebins[i]);
  costheta_analysis.reserve(n_costh_analysis + 1);
  for (size_t i = 0; i <= n_costh_analysis * costh_rebin_factor; i += costh_rebin_factor)
    costheta_analysis.push_back(costheta_bins[i]);
  UpdatePrediction();
}

void BinnedInteraction::ensure_pod_flux_xsec() const {
  if (pod_flux_valid) return;
  flux_pod_numu    = PodHist2D<oscillaton_calc_precision>::from_th2d(flux_hist_numu);
  flux_pod_numubar = PodHist2D<oscillaton_calc_precision>::from_th2d(flux_hist_numubar);
  flux_pod_nue     = PodHist2D<oscillaton_calc_precision>::from_th2d(flux_hist_nue);
  flux_pod_nuebar  = PodHist2D<oscillaton_calc_precision>::from_th2d(flux_hist_nuebar);
  xsec_pod_numu    = th1d_to_pod(xsec_hist_numu);
  xsec_pod_numubar = th1d_to_pod(xsec_hist_numubar);
  xsec_pod_nue     = th1d_to_pod(xsec_hist_nue);
  xsec_pod_nuebar  = th1d_to_pod(xsec_hist_nuebar);
  pod_flux_valid = true;
}

void BinnedInteraction::UpdatePrediction() {
  ensure_pod_flux_xsec();

  // POD probability histograms (no TH2D intermediates).
  auto podP = propagator->GetProb_Hists_POD(Ebins, costheta_bins, *this);

  // Size prediction arrays to analysis binning.
  pred_pod_numu    = PodHist2D<double>(n_costh_analysis, n_e_analysis);
  pred_pod_numubar = PodHist2D<double>(n_costh_analysis, n_e_analysis);
  pred_pod_nue     = PodHist2D<double>(n_costh_analysis, n_e_analysis);
  pred_pod_nuebar  = PodHist2D<double>(n_costh_analysis, n_e_analysis);

  // Compute oscillated prediction: loop over fine bins, rebin to analysis.
  // Layout: podP[nu][from][to](costh, E)  — same as TH2D version.
#pragma omp parallel for collapse(2)
  for (size_t cA = 0; cA < n_costh_analysis; ++cA) {
    for (size_t eA = 0; eA < n_e_analysis; ++eA) {
      double sum_numu = 0, sum_numubar = 0, sum_nue = 0, sum_nuebar = 0;

      const size_t c_start = cA * costh_rebin_factor;
      const size_t c_end   = c_start + costh_rebin_factor;
      const size_t e_start = eA * E_rebin_factor;
      const size_t e_end   = e_start + E_rebin_factor;

      for (size_t c = c_start; c < c_end; ++c) {
        for (size_t e = e_start; e < e_end; ++e) {
          const auto fn = flux_pod_nue(c, e);
          const auto fm = flux_pod_numu(c, e);
          const auto fnb = flux_pod_nuebar(c, e);
          const auto fmb = flux_pod_numubar(c, e);

          sum_numu    += (podP[0][0][1](c, e) * fn + podP[0][1][1](c, e) * fm) * xsec_pod_numu[e];
          sum_nue     += (podP[0][0][0](c, e) * fn + podP[0][1][0](c, e) * fm) * xsec_pod_nue[e];
          sum_numubar += (podP[1][0][1](c, e) * fnb + podP[1][1][1](c, e) * fmb) * xsec_pod_numubar[e];
          sum_nuebar  += (podP[1][0][0](c, e) * fnb + podP[1][1][0](c, e) * fmb) * xsec_pod_nuebar[e];
        }
      }

      pred_pod_numu(cA, eA)    = sum_numu;
      pred_pod_numubar(cA, eA) = sum_numubar;
      pred_pod_nue(cA, eA)     = sum_nue;
      pred_pod_nuebar(cA, eA)  = sum_nuebar;
    }
  }
}

void BinnedInteraction::proposeStep() {
  OscillationParameters::proposeStep();
  UpdatePrediction();
}

double
BinnedInteraction::GetLogLikelihoodAgainstData(const SimpleDataHist &dataset) const {
  auto chi2_numu    = pod_chi2(dataset.data_numu,    pred_pod_numu);
  auto chi2_numubar = pod_chi2(dataset.data_numubar, pred_pod_numubar);
  auto chi2_nue     = pod_chi2(dataset.data_nue,     pred_pod_nue);
  auto chi2_nuebar  = pod_chi2(dataset.data_nuebar,  pred_pod_nuebar);
  return -0.5 * (chi2_numu + chi2_numubar + chi2_nue + chi2_nuebar);
}

SimpleDataHist BinnedInteraction::GenerateData() const {
  SimpleDataHist data;
  data.data_numu    = pred_pod_numu;
  data.data_numubar = pred_pod_numubar;
  data.data_nue     = pred_pod_nue;
  data.data_nuebar  = pred_pod_nuebar;
  data.Ebins         = Ebins_analysis;
  data.costheta_bins = costheta_analysis;
  return data;
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
  // Write prediction from POD, using analysis bin edges.
  auto pn = pred_pod_numu.to_th2d(Ebins_analysis, costheta_analysis, "Prediction_hist_numu");
  auto pnb = pred_pod_numubar.to_th2d(Ebins_analysis, costheta_analysis, "Prediction_hist_numubar");
  auto pne = pred_pod_nue.to_th2d(Ebins_analysis, costheta_analysis, "Prediction_hist_nue");
  auto pneb = pred_pod_nuebar.to_th2d(Ebins_analysis, costheta_analysis, "Prediction_hist_nuebar");
  pn.Write(); pnb.Write(); pne.Write(); pneb.Write();
  file->Write();
  file->Close();
  delete file;
}

SimpleDataHist BinnedInteraction::GenerateData_NoOsc() const {
  ensure_pod_flux_xsec();
  PodHist2D<double> noosc_numu(n_costh_fine, n_e_fine);
  PodHist2D<double> noosc_numubar(n_costh_fine, n_e_fine);
  PodHist2D<double> noosc_nue(n_costh_fine, n_e_fine);
  PodHist2D<double> noosc_nuebar(n_costh_fine, n_e_fine);

#pragma omp parallel for collapse(2)
  for (size_t c = 0; c < n_costh_fine; ++c)
    for (size_t e = 0; e < n_e_fine; ++e) {
      noosc_numu(c, e)    = flux_pod_numu(c, e)    * xsec_pod_numu[e];
      noosc_numubar(c, e) = flux_pod_numubar(c, e) * xsec_pod_numubar[e];
      noosc_nue(c, e)     = flux_pod_nue(c, e)     * xsec_pod_nue[e];
      noosc_nuebar(c, e)  = flux_pod_nuebar(c, e)  * xsec_pod_nuebar[e];
    }

  SimpleDataHist data;
  data.data_numu    = noosc_numu;
  data.data_numubar = noosc_numubar;
  data.data_nue     = noosc_nue;
  data.data_nuebar  = noosc_nuebar;
  data.Ebins         = Ebins;
  data.costheta_bins = costheta_bins;
  return data;
}

double BinnedInteraction::GetLogLikelihood() const {
  auto llh = OscillationParameters::GetLogLikelihood();
  if (GetDM32sq() < 0)
    llh += log_ih_bias;
  return llh;
}

void BinnedInteraction::Save_prob_hist(const std::string &name) {
  auto file = TFile::Open(name.c_str(), "RECREATE");
  file->cd();
  auto pod = propagator->GetProb_Hists_3F_POD(Ebins, costheta_bins, *this);
  auto id_2_name = std::to_array({"nue", "numu", "nutau"});
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        auto h = pod[i][j][k].to_th2d(Ebins, costheta_bins);
        h.SetName(std::format("{}_{}_{}", i == 0 ? "neutrino" : "antineutrino",
                              id_2_name[j], id_2_name[k])
                      .c_str());
        h.Write();
      }
  file->Close();
  delete file;
}
