// All BinnedInteraction methods except the production constructor.
// The production constructor lives in BinnedInteractionProd.cxx and depends on
// the HondaFlux2D and GENIE_XSEC global objects.  Keeping them separate lets
// pybind (and tests) link this TU without the physics-data-file globals.

#include "BinnedInteraction.h"
#include "OscillationParameters.h"

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
    : propagator{std::move(prop)} {
  auto imm = std::make_shared<BinnedInteractionImmutable>();
  imm->Ebins = std::move(Ebins_);
  imm->costheta_bins = std::move(costheta_bins_);
  imm->E_rebin_factor = E_rebin_factor_;
  imm->costh_rebin_factor = costh_rebin_factor_;
  imm->n_costh_fine = imm->costheta_bins.size() - 1;
  imm->n_e_fine = imm->Ebins.size() - 1;
  imm->n_costh_analysis = imm->n_costh_fine / costh_rebin_factor_;
  imm->n_e_analysis = imm->n_e_fine / E_rebin_factor_;
  imm->flux_numu    = std::move(histos.pod_flux_numu);
  imm->flux_numubar = std::move(histos.pod_flux_numubar);
  imm->flux_nue     = std::move(histos.pod_flux_nue);
  imm->flux_nuebar  = std::move(histos.pod_flux_nuebar);
  imm->xsec_numu    = std::move(histos.pod_xsec_numu);
  imm->xsec_numubar = std::move(histos.pod_xsec_numubar);
  imm->xsec_nue     = std::move(histos.pod_xsec_nue);
  imm->xsec_nuebar  = std::move(histos.pod_xsec_nuebar);
  imm->log_ih_bias  = std::log(IH_bias_);

  imm->Ebins_analysis.reserve(imm->n_e_analysis + 1);
  for (size_t i = 0; i <= imm->n_e_analysis * imm->E_rebin_factor; i += imm->E_rebin_factor)
    imm->Ebins_analysis.push_back(imm->Ebins[i]);
  imm->costheta_analysis.reserve(imm->n_costh_analysis + 1);
  for (size_t i = 0; i <= imm->n_costh_analysis * imm->costh_rebin_factor; i += imm->costh_rebin_factor)
    imm->costheta_analysis.push_back(imm->costheta_bins[i]);

  imm_ = std::move(imm);
  UpdatePrediction();
}

void BinnedInteraction::UpdatePrediction() {

  propagator->re_calculate(*this);

  // Size prediction arrays to analysis binning.
  pred_pod_numu    = PodHist2D<double>(imm_->n_costh_analysis, imm_->n_e_analysis);
  pred_pod_numubar = PodHist2D<double>(imm_->n_costh_analysis, imm_->n_e_analysis);
  pred_pod_nue     = PodHist2D<double>(imm_->n_costh_analysis, imm_->n_e_analysis);
  pred_pod_nuebar  = PodHist2D<double>(imm_->n_costh_analysis, imm_->n_e_analysis);

  const auto nCA = imm_->n_costh_analysis, nEA = imm_->n_e_analysis;
  const auto cRF = imm_->costh_rebin_factor, eRF = imm_->E_rebin_factor;

  // Raw-access fast path: read oscillation probabilities directly from the
  // calculator's result buffer (pinned host memory), eliminating the 8 ×
  // PodHist2D<double> temporary allocation (~6.4 MB at fine binning).
  if (propagator->has_raw_results()) {
    const auto *raw_nu  = propagator->raw_prob_neutrino();
    const auto *raw_anu = propagator->raw_prob_antineutrino();
    const size_t nC = propagator->raw_n_cosines();
    const size_t nE = propagator->raw_n_energies();

    // CUDAProb3 layout: [ProbType * nC * nE + cos * nE + e]
    // ProbType: e_e=0, e_m=1, m_e=3, m_m=4
    const size_t off_ee = 0 * nC * nE, off_em = 1 * nC * nE;
    const size_t off_me = 3 * nC * nE, off_mm = 4 * nC * nE;

#pragma omp parallel for collapse(2)
    for (size_t cA = 0; cA < nCA; ++cA) {
      for (size_t eA = 0; eA < nEA; ++eA) {
        double sum_numu = 0, sum_numubar = 0, sum_nue = 0, sum_nuebar = 0;

        const size_t c_start = cA * cRF;
        const size_t c_end   = c_start + cRF;
        const size_t e_start = eA * eRF;
        const size_t e_end   = e_start + eRF;

        for (size_t c = c_start; c < c_end; ++c) {
          for (size_t e = e_start; e < e_end; ++e) {
            const auto fn = imm_->flux_nue(c, e);
            const auto fm = imm_->flux_numu(c, e);
            const auto fnb = imm_->flux_nuebar(c, e);
            const auto fmb = imm_->flux_numubar(c, e);

            const size_t idx = c * nE + e;

            // Neutrino: P(nue→numu)=e_m, P(numu→numu)=m_m, P(nue→nue)=e_e, P(numu→nue)=m_e
            const auto pem = raw_nu[off_em + idx], pmm = raw_nu[off_mm + idx];
            const auto pee = raw_nu[off_ee + idx], pme = raw_nu[off_me + idx];

            // Antineutrino
            const auto pa_em = raw_anu[off_em + idx], pa_mm = raw_anu[off_mm + idx];
            const auto pa_ee = raw_anu[off_ee + idx], pa_me = raw_anu[off_me + idx];

            sum_numu    += (pem * fn   + pmm * fm)   * imm_->xsec_numu[e];
            sum_nue     += (pee * fn   + pme * fm)   * imm_->xsec_nue[e];
            sum_numubar += (pa_em * fnb + pa_mm * fmb) * imm_->xsec_numubar[e];
            sum_nuebar  += (pa_ee * fnb + pa_me * fmb) * imm_->xsec_nuebar[e];
          }
        }

        pred_pod_numu(cA, eA)    = sum_numu;
        pred_pod_numubar(cA, eA) = sum_numubar;
        pred_pod_nue(cA, eA)     = sum_nue;
        pred_pod_nuebar(cA, eA)  = sum_nuebar;
      }
    }
  } else {
    // Fallback for propagators that don't provide raw results (e.g. test mocks).
    auto podP = propagator->GetProb_Hists_POD(imm_->Ebins, imm_->costheta_bins, *this);

#pragma omp parallel for collapse(2)
    for (size_t cA = 0; cA < nCA; ++cA) {
      for (size_t eA = 0; eA < nEA; ++eA) {
        double sum_numu = 0, sum_numubar = 0, sum_nue = 0, sum_nuebar = 0;

        const size_t c_start = cA * cRF;
        const size_t c_end   = c_start + cRF;
        const size_t e_start = eA * eRF;
        const size_t e_end   = e_start + eRF;

        for (size_t c = c_start; c < c_end; ++c) {
          for (size_t e = e_start; e < e_end; ++e) {
            const auto fn = imm_->flux_nue(c, e);
            const auto fm = imm_->flux_numu(c, e);
            const auto fnb = imm_->flux_nuebar(c, e);
            const auto fmb = imm_->flux_numubar(c, e);

            sum_numu    += (podP[0][0][1](c, e) * fn + podP[0][1][1](c, e) * fm) * imm_->xsec_numu[e];
            sum_nue     += (podP[0][0][0](c, e) * fn + podP[0][1][0](c, e) * fm) * imm_->xsec_nue[e];
            sum_numubar += (podP[1][0][1](c, e) * fnb + podP[1][1][1](c, e) * fmb) * imm_->xsec_numubar[e];
            sum_nuebar  += (podP[1][0][0](c, e) * fnb + podP[1][1][0](c, e) * fmb) * imm_->xsec_nuebar[e];
          }
        }

        pred_pod_numu(cA, eA)    = sum_numu;
        pred_pod_numubar(cA, eA) = sum_numubar;
        pred_pod_nue(cA, eA)     = sum_nue;
        pred_pod_nuebar(cA, eA)  = sum_nuebar;
      }
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
  data.Ebins         = imm_->Ebins_analysis;
  data.costheta_bins = imm_->costheta_analysis;
  return data;
}

void BinnedInteraction::SaveAs(const char *filename) const {
  auto file = TFile::Open(filename, "RECREATE");
  file->cd();
  imm_->flux_numu.to_th2d(imm_->Ebins, imm_->costheta_bins, "flux_hist_numu").Write();
  imm_->flux_numubar.to_th2d(imm_->Ebins, imm_->costheta_bins, "flux_hist_numubar").Write();
  imm_->flux_nue.to_th2d(imm_->Ebins, imm_->costheta_bins, "flux_hist_nue").Write();
  imm_->flux_nuebar.to_th2d(imm_->Ebins, imm_->costheta_bins, "flux_hist_nuebar").Write();
  for (auto &[pod, name] : {
           std::tuple{&imm_->xsec_numu,    "xsec_hist_numu"},
           std::tuple{&imm_->xsec_numubar, "xsec_hist_numubar"},
           std::tuple{&imm_->xsec_nue,     "xsec_hist_nue"},
           std::tuple{&imm_->xsec_nuebar,  "xsec_hist_nuebar"}}) {
    TH1D h(name, "", static_cast<int>(imm_->Ebins.size()) - 1, imm_->Ebins.data());
    for (size_t i = 0; i < pod->size(); ++i)
      h.SetBinContent(static_cast<int>(i) + 1, (*pod)[i]);
    h.Write();
  }
  pred_pod_numu.to_th2d(imm_->Ebins_analysis, imm_->costheta_analysis, "Prediction_hist_numu").Write();
  pred_pod_numubar.to_th2d(imm_->Ebins_analysis, imm_->costheta_analysis, "Prediction_hist_numubar").Write();
  pred_pod_nue.to_th2d(imm_->Ebins_analysis, imm_->costheta_analysis, "Prediction_hist_nue").Write();
  pred_pod_nuebar.to_th2d(imm_->Ebins_analysis, imm_->costheta_analysis, "Prediction_hist_nuebar").Write();
  file->Write();
  file->Close();
  delete file;
}

SimpleDataHist BinnedInteraction::GenerateData_NoOsc() const {
  const auto nCF = imm_->n_costh_fine, nEF = imm_->n_e_fine;
  PodHist2D<double> noosc_numu(nCF, nEF);
  PodHist2D<double> noosc_numubar(nCF, nEF);
  PodHist2D<double> noosc_nue(nCF, nEF);
  PodHist2D<double> noosc_nuebar(nCF, nEF);

#pragma omp parallel for collapse(2)
  for (size_t c = 0; c < nCF; ++c)
    for (size_t e = 0; e < nEF; ++e) {
      noosc_numu(c, e)    = imm_->flux_numu(c, e)    * imm_->xsec_numu[e];
      noosc_numubar(c, e) = imm_->flux_numubar(c, e) * imm_->xsec_numubar[e];
      noosc_nue(c, e)     = imm_->flux_nue(c, e)     * imm_->xsec_nue[e];
      noosc_nuebar(c, e)  = imm_->flux_nuebar(c, e)  * imm_->xsec_nuebar[e];
    }

  SimpleDataHist data;
  data.data_numu    = noosc_numu;
  data.data_numubar = noosc_numubar;
  data.data_nue     = noosc_nue;
  data.data_nuebar  = noosc_nuebar;
  data.Ebins         = imm_->Ebins;
  data.costheta_bins = imm_->costheta_bins;
  return data;
}

double BinnedInteraction::GetLogLikelihood() const {
  auto llh = OscillationParameters::GetLogLikelihood();
  if (GetDM32sq() < 0)
    llh += imm_->log_ih_bias;
  return llh;
}

void BinnedInteraction::Save_prob_hist(const std::string &name) {
  auto file = TFile::Open(name.c_str(), "RECREATE");
  file->cd();
  auto pod = propagator->GetProb_Hists_3F_POD(imm_->Ebins, imm_->costheta_bins, *this);
  auto id_2_name = std::to_array({"nue", "numu", "nutau"});
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        auto h = pod[i][j][k].to_th2d(imm_->Ebins, imm_->costheta_bins);
        h.SetName(std::format("{}_{}_{}", i == 0 ? "neutrino" : "antineutrino",
                              id_2_name[j], id_2_name[k])
                      .c_str());
        h.Write();
      }
  file->Close();
  delete file;
}
