#include "BinnedInteraction.h"
#include "OscillationParameters.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
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
#include <ostream>
#include <print>
#include <ranges>

class MinuitFitter final : public ROOT::Minuit2::FCNBase {
public:
  MinuitFitter(BinnedInteraction &binned_interaction_, SimpleDataHist &data_,
               pull_toggle toggle_ = all_on)
      : binned_interaction(binned_interaction_), data(data_), toggle(toggle_) {}

  double operator()(const std::vector<double> &params) const override {
    binned_interaction.set_param({.DM2 = params[0],
                                  .Dm2 = params[1],
                                  .T23 = params[2],
                                  .T13 = params[3],
                                  .T12 = params[4],
                                  .DCP = params[5]});
    binned_interaction.UpdatePrediction();
    auto llh = binned_interaction.GetLogLikelihoodAgainstData(data);
    llh += binned_interaction.OscillationParameters::GetLogLikelihood(toggle);
    // std::println("llh: {}", -2 * llh);
    return -2. * llh;
  }

  double Up() const override { return 1.; }

private:
  mutable BinnedInteraction binned_interaction;
  SimpleDataHist data;
  pull_toggle toggle;
};

int main(int argc, char **agrv) {
  // gSystem->Lo
  gErrorIgnoreLevel = kWarning;
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat("4.1f");
  TH1::AddDirectory(false);
  auto costheta_bins = linspace(-1., 1., 481);

  auto Ebins = logspace(0.1, 20., 401);

  BinnedInteraction interaction_model_NH{
      Ebins, costheta_bins, scale_factor_1y, 40, 40, 1};
  auto asimov_data_NH_1y = interaction_model_NH.GenerateData(); // data for NH

  BinnedInteraction interaction_model_IH{
      Ebins, costheta_bins, scale_factor_1y, 40, 40, 1};
  interaction_model_IH.flip_hierarchy();
  interaction_model_IH.UpdatePrediction();
  auto asimov_data_IH_1y = interaction_model_IH.GenerateData();

  // auto asimov_data_IH_6y = asimov_data_IH_1y;
  // asimov_data_IH_6y.Scale(6);
  // asimov_data_IH_6y.SaveAs("asimov_data_IH_6y.root");
  // return 0;
  // std::println(std::cout, "Mock chi2: {}",
  //              TH2D_chi2(cdata.hist_numu, cdata.hist_numubar));

  auto do_fit_and_plot = [&](double dm32_init,
                             const pull_toggle &toggles = all_on,
                             const double n_years = 6) {
    BinnedInteraction interaction_model_fit{
        Ebins, costheta_bins, scale_factor_1y * n_years, 40, 40, 1};

    auto data_to_fit = dm32_init > 0 ? asimov_data_IH_1y : asimov_data_NH_1y;
    data_to_fit.Scale(n_years);
    MinuitFitter fitter_chi2(interaction_model_fit, data_to_fit, toggles);
    auto tag = dm32_init > 0 ? "NH" : "IH";

    std::println("{:*^25}", tag);

    {
      auto enabled_pull = toggles.get_active();
      auto disabled_pull = toggles.get_inactive();
      if (!enabled_pull.empty()) {
        std::print("enabled pull: ");
        for (auto &name : enabled_pull) {
          std::cout << name << ' ';
        }
      }
      if (!disabled_pull.empty()) {
        std::print("\ndisabled pull: ");
        for (auto &name : disabled_pull) {
          std::cout << name << ' ';
        }
      }
      std::cout << '\n';
    }

    gSystem->MakeDirectory(tag);
    gSystem->ChangeDirectory(tag);
    constexpr double dm32_limit_abs = 1e-1;
    double dm32_min = dm32_init > 0 ? 0 : -dm32_limit_abs;
    double dm32_max = dm32_init > 0 ? dm32_limit_abs : 0;

    auto get_random = [&]() { return gRandom->Uniform(0, 2.0); };

    ROOT::Minuit2::MnUserParameters param{};
    param.Add("#Delta M_{32}^{2}", dm32_init * get_random(), 0.01e-3, dm32_min,
              dm32_max);
    param.Add("#Delta M_{21}^{2}", 7.538905177e-05 * get_random(), 0.01e-4, 0,
              1e-3);
    param.Add("sin^{2}#theta_{23}", 0.5 * get_random(), 0.01, 0, 1);
    param.Add("sin^{2}#theta_{13}", 0.5 * get_random(), 0.01e-1, 0, 1);
    param.Add("sin^{2}#theta_{12}", 0.5 * get_random(), 0.01, 0, 1);
    param.Add("#delta_{CP}", 3.639202885 * get_random(), 0.1, 0, 2 * M_PI);

    interaction_model_fit.set_param({.DM2 = param.Value(0),
                                     .Dm2 = param.Value(1),
                                     .T23 = param.Value(2),
                                     .T13 = param.Value(3),
                                     .T12 = param.Value(4),
                                     .DCP = param.Value(5)});
    interaction_model_fit.UpdatePrediction();
    auto pre_fit = interaction_model_fit.GenerateData();
    std::println(
        "chi2 pre-fit: {:.4f}",
        -2 * interaction_model_fit.GetLogLikelihoodAgainstData(data_to_fit));

    const auto result = ROOT::Minuit2::MnMigrad{fitter_chi2, param}();

    if (!result.HasValidParameters()) {
      std::println("!!! Warning: HasValidParameters -> false");
    }
    if (!result.IsValid()) {
      std::println("!!! Warning: IsValid -> false");
    }

    const auto &final_params = result.UserParameters();
    // for (size_t i = 0; i < final_params.Params().size(); ++i) {
    //   std::println("{}: {:2f}", final_params.GetName(i),
    //   final_params.Value(i));
    // }

    std::cout << final_params << '\n';

    auto minimal_chi2 = result.Fval();

    std::println("Fval: {:.4e}", minimal_chi2);

    interaction_model_fit.set_param({.DM2 = final_params.Value(0),
                                     .Dm2 = final_params.Value(1),
                                     .T23 = final_params.Value(2),
                                     .T13 = final_params.Value(3),
                                     .T12 = final_params.Value(4),
                                     .DCP = final_params.Value(5)});

    interaction_model_fit.UpdatePrediction();

    auto llh_to_data =
        interaction_model_fit.GetLogLikelihoodAgainstData(data_to_fit);
    auto pull = interaction_model_fit.GetLogLikelihood();

    std::println("chi2 {:.4e}, data: {:.4e}, pull: {:.4e}",
                 -2 * (llh_to_data + pull), -2 * llh_to_data, -2 * pull);

    std::println("{:*^25}\n\n", "finished");
    std::flush(std::cout);
    gSystem->ChangeDirectory("..");
    return result.Fval();
  };
  // do_fit_and_plot(-4.91e-3, SK_w_T13, 6);

  std::ranges::iota_view iter_times{0, 12};
  std::println("SK_w_T13 true NH: min chi2: {:.4e}",
               std::ranges::min(iter_times | std::views::transform([&](int) {
                                  return do_fit_and_plot(-4.91e-3, SK_w_T13, 6);
                                })));
  std::println("SK_w_T13 true IH: min chi2: {:.4e}",
               std::ranges::min(iter_times | std::views::transform([&](int) {
                                  return do_fit_and_plot(4.91e-3, SK_w_T13, 6);
                                })));
  std::println("SK_wo_T13 true NH: min chi2: {:.4e}",
               std::ranges::min(iter_times | std::views::transform([&](int) {
                                  return do_fit_and_plot(-4.91e-3, SK_wo_T13,
                                                         6);
                                })));
  std::println("SK_wo_T13 true IH: min chi2: {:.4e}",
               std::ranges::min(iter_times | std::views::transform([&](int) {
                                  return do_fit_and_plot(4.91e-3, SK_wo_T13, 6);
                                })));

  for (int i = 0; i < 6; ++i) {
    pull_toggle one_off = all_on;
    one_off[i] = false;
    // do_fit_and_plot(-4.91e-3, one_off);
    std::println("one_off, true NH min chi2: {:.4e}",
                 std::ranges::min(iter_times | std::views::transform([&](int) {
                                    return do_fit_and_plot(-4.91e-3, one_off, 6);
                                  })));
    std::println("one_off, true IH min chi2: {:.4e}",
                 std::ranges::min(iter_times | std::views::transform([&](int) {
                                    return do_fit_and_plot(4.91e-3, one_off, 6);
                                  })));
    pull_toggle one_on = all_off;
    one_on[i] = true;
    // do_fit_and_plot(-4.91e-3, one_on);
    // do_fit_and_plot(4.91e-3, one_on);
    std::println("one_on, true NH min chi2: {:.4e}",
                 std::ranges::min(iter_times | std::views::transform([&](int) {
                                    return do_fit_and_plot(-4.91e-3, one_on, 6);
                                  })));
    std::println("one_on, true IH min chi2: {:.4e}",
                 std::ranges::min(iter_times | std::views::transform([&](int) {
                                    return do_fit_and_plot(4.91e-3, one_on, 6);
                                  })));
  }

  return 0;
}
