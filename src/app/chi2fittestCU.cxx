#include "OscillationParameters.h"
#include "ParBinnedInterface.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "fit_config.h"
#include "tools.h"

#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnContours.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnUserParameters.h>

#include <TCanvas.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TMath.h>
#include <TPad.h>
#include <TRandom.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TVirtualPad.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <omp.h>
#include <print>
#include <ranges>

class MinuitFitter final : public ROOT::Minuit2::FCNBase {
public:
  MinuitFitter(ParBinnedInterface &binned_interaction_, SimpleDataHist &data_)
      : binned_interaction(binned_interaction_), data(data_) {}

  double operator()(const std::vector<double> &params) const override {
    binned_interaction.set_param({.DM2 = params[0],
                                  .Dm2 = params[1],
                                  .T23 = params[2],
                                  .T13 = params[3],
                                  .T12 = params[4],
                                  .DCP = params[5]});
    binned_interaction.UpdatePrediction();
    auto llh = binned_interaction.GetLogLikelihoodAgainstData(data);
    llh += binned_interaction.GetLogLikelihood();
    std::cout << llh << std::endl;
    return -2. * llh;
  }

  double Up() const override { return 1.; }

private:
  mutable ParBinnedInterface binned_interaction;
  SimpleDataHist data;
};

namespace {
SimpleDataHist operator/(const SimpleDataHist &lhs, const SimpleDataHist &rhs) {
  SimpleDataHist ret;
  ret.Ebins = lhs.Ebins;
  ret.costheta_bins = lhs.costheta_bins;

  auto divide_pod = [](const auto &a, const auto &b) {
    PodHist2D<double> result(a.n_costh, a.n_e);
    for (size_t i = 0; i < a.size(); ++i)
      result.data[i] = a.data[i] / b.data[i];
    return result;
  };

  ret.data_numu    = divide_pod(lhs.data_numu,    rhs.data_numu);
  ret.data_numubar = divide_pod(lhs.data_numubar, rhs.data_numubar);
  ret.data_nue     = divide_pod(lhs.data_nue,     rhs.data_nue);
  ret.data_nuebar  = divide_pod(lhs.data_nuebar,  rhs.data_nuebar);
  return ret;
}

double chi2_possion(double bin_data, double bin_pred) {
  return ((bin_pred - bin_data) + bin_data * log(bin_data / bin_pred)) * 2;
}

PodHist2D<double> get_chi2_hist(const PodHist2D<double> &data,
                                const PodHist2D<double> &pred) {
  PodHist2D<double> ret(data.n_costh, data.n_e);
  for (size_t i = 0; i < data.size(); ++i) {
    const double d = data.data[i];
    const double p = pred.data[i];
    if (d != 0)
      ret.data[i] = chi2_possion(d, p);
    else
      ret.data[i] = p;
  }
  return ret;
}

SimpleDataHist get_chi2_data(const SimpleDataHist &data,
                             const SimpleDataHist &pred) {
  SimpleDataHist ret;
  ret.Ebins = data.Ebins;
  ret.costheta_bins = data.costheta_bins;
  ret.data_numu    = get_chi2_hist(data.data_numu,    pred.data_numu);
  ret.data_numubar = get_chi2_hist(data.data_numubar, pred.data_numubar);
  ret.data_nue     = get_chi2_hist(data.data_nue,     pred.data_nue);
  ret.data_nuebar  = get_chi2_hist(data.data_nuebar,  pred.data_nuebar);
  return ret;
}

} // namespace

template <class T>
void plot_data(T &&data, const std::string &filename, double min = 0.,
               double max = 0.) {
  auto reset_style = [&](TH1 &hist, const std::string &title) {
    // gPad->SetTopMargin(gPad->GetTopMargin() / 2.);
    gPad->SetBottomMargin(gPad->GetBottomMargin() * 1.5);
    gPad->SetRightMargin(gPad->GetRightMargin() * 1.15);
    ResetStyle(&hist);
    hist.SetTitle(title.c_str());
    hist.GetXaxis()->SetTitle("E_{#nu} (GeV)");
    hist.GetYaxis()->SetTitle("cos#it{#theta}");
    hist.GetXaxis()->CenterTitle();
    hist.GetYaxis()->CenterTitle();
    if (min != 0) {
      hist.SetMinimum(min);
    }
    if (max != 0) {
      hist.SetMaximum(max);
    }
  };
  TCanvas c1{};
  c1.SetLogx();
  c1.Divide(2, 2);
  c1.cd(1)->SetLogx();
  auto h_numu = data.hist_numu();
  reset_style(h_numu, "#nu_{#mu}");
  h_numu.Draw("COLZ TEXT");
  c1.cd(2)->SetLogx();
  auto h_numubar = data.hist_numubar();
  reset_style(h_numubar, "#bar{#nu}_{#mu}");
  h_numubar.Draw("COLZ TEXT");
  c1.cd(3)->SetLogx();
  auto h_nue = data.hist_nue();
  reset_style(h_nue, "#nu_{e}");
  h_nue.Draw("COLZ TEXT");
  c1.cd(4)->SetLogx();
  auto h_nuebar = data.hist_nuebar();
  reset_style(h_nuebar, "#bar{#nu}_{e}");
  h_nuebar.Draw("COLZ TEXT");
  c1.SaveAs(filename.c_str());
}

int main(int argc, char **agrv) {
  // gSystem->Lo
  gErrorIgnoreLevel = kWarning;
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat("4.1f");
  TH1::AddDirectory(false);
  auto costheta_bins = linspace(-1., 1., FitConfig::n_costheta_bins + 1);
  auto Ebins = logspace(FitConfig::e_min, FitConfig::e_max, FitConfig::n_energy_bins + 1);

  ParBinnedInterface bint{Ebins, costheta_bins, FitConfig::scale_factor,
                           FitConfig::E_rebin_factor, FitConfig::costh_rebin_factor, 1};
  auto cdata_NH = bint.GenerateData(); // data for NH

  // ParBinnedInterface bint_1{Ebins, costheta_bins, scale_factor, 40, 40, 1};
  bint.flip_hierarchy();
  auto cdata_IH = bint.GenerateData(); // data for NH

  plot_data(cdata_NH, "asimovNH.pdf");
  plot_data(cdata_IH, "asimovIH.pdf");

  auto do_fit_and_plot = [&](double dm32_init,
                             const pull_toggle &toggles = all_on,
                             const double n_years = 6) -> double {
    bint.set_toggle(toggles);
    auto &data_to_fit = dm32_init > 0 ? cdata_IH : cdata_NH;
    MinuitFitter fitter_chi2(bint, data_to_fit);
    auto tag = dm32_init > 0 ? "NH" : "IH";
    std::println("{:*^15}", tag);

    {
      auto enabled_pull = toggles.get_active();
      auto disabled_pull = toggles.get_inactive();
      if (!enabled_pull.empty()) {
        std::print("\n\nenabled pull: ");
        for (auto &name : enabled_pull) {
          std::cout << name << ' ';
        }
      }
      if (!disabled_pull.empty()) {
        std::print("\n\ndisabled pull: ");
        for (auto &name : disabled_pull) {
          std::cout << name << ' ';
        }
      }
      std::cout << '\n';
    }

    gSystem->MakeDirectory(tag);
    gSystem->ChangeDirectory(tag);

    double dm32_min = dm32_init > 0 ? 0 : -1;
    double dm32_max = dm32_init > 0 ? 1 : 0;

    auto get_random = [&]() { return gRandom->Uniform(0.2, 1.8); };

    ROOT::Minuit2::MnUserParameters param{};
    param.Add("#Delta M_{32}^{2}", dm32_init * get_random(), 0.001e-3, dm32_min,
              dm32_max);
    param.Add("#Delta M_{21}^{2}", 1.56e-4 * get_random(), 0.01e-5, 0, 1);
    param.Add("sin^{2}#theta_{23}", 0.75 * get_random(), 0.001, 0.5, 1);
    param.Add("sin^{2}#theta_{13}", 0.0439 * get_random(), 0.01e-2, 0, 1);
    param.Add("sin^{2}#theta_{12}", 0.614 * get_random(), 0.001, 0, 1);
    param.Add("#delta_{CP}", 0.59 * M_PI * get_random(), 0.01, 0, 2 * M_PI);

    bint.set_param({.DM2 = param.Value(0),
                    .Dm2 = param.Value(1),
                    .T23 = param.Value(2),
                    .T13 = param.Value(3),
                    .T12 = param.Value(4),
                    .DCP = param.Value(5)});
    bint.UpdatePrediction();

    auto result = ROOT::Minuit2::MnMigrad{fitter_chi2, param}();

    if (!result.HasValidParameters()) {
      std::println("failed: {}", tag);
      return INFINITY;
    }

    const auto &final_params = result.UserParameters();
    for (size_t i = 0; i < final_params.Params().size(); ++i) {
      std::println("{}: {:2f}", final_params.GetName(i), final_params.Value(i));
    }

    auto minimal_chi2 = result.Fval();

    std::println("Fval: {:.4e}", minimal_chi2);

    bint.set_param({.DM2 = final_params.Value(0),
                    .Dm2 = final_params.Value(1),
                    .T23 = final_params.Value(2),
                    .T13 = final_params.Value(3),
                    .T12 = final_params.Value(4),
                    .DCP = final_params.Value(5)});

    bint.UpdatePrediction();
    auto llh_to_data = bint.GetLogLikelihoodAgainstData(cdata_NH);
    auto pull = bint.GetLogLikelihood();

    std::println("chi2 {:.4e}, data: {:.4e}, pull: {:.4e}",
                 -2 * (llh_to_data + pull), -2 * llh_to_data, -2 * pull);

    gSystem->ChangeDirectory("..");
    std::println("{:*^15}", "finished");

    // return result.Fval();
    if (std::isnan(minimal_chi2)) {
      return INFINITY;
    }
    return minimal_chi2;
  };
  do_fit_and_plot(4e-3);
  // for (int i = 0; i < 6; ++i) {
  //   pull_toggle one_off = all_on;
  //   one_off[i] = false;
  //   // do_fit_and_plot(-4.91e-3, one_off);
  //   std::println("one_off, true NH min chi2: {:.4e}",
  //                std::ranges::min(iter_times | std::views::transform([&](int) {
  //                                   return do_fit_and_plot(-4.91e-3, one_off,
  //                                                          6);
  //                                 })));
  //   std::println("one_off, true IH min chi2: {:.4e}",
  //                std::ranges::min(iter_times | std::views::transform([&](int) {
  //                                   return do_fit_and_plot(4.91e-3, one_off, 6);
  //                                 })));
  //   pull_toggle one_on = all_off;
  //   one_on[i] = true;
  //   // do_fit_and_plot(-4.91e-3, one_on);
  //   // do_fit_and_plot(4.91e-3, one_on);
  //   std::println("one_on, true NH min chi2: {:.4e}",
  //                std::ranges::min(iter_times | std::views::transform([&](int) {
  //                                   return do_fit_and_plot(-4.91e-3, one_on, 6);
  //                                 })));
  //   std::println("one_on, true IH min chi2: {:.4e}",
  //                std::ranges::min(iter_times | std::views::transform([&](int) {
  //                                   return do_fit_and_plot(4.91e-3, one_on, 6);
  //                                 })));
  // }
  return 0;
}
