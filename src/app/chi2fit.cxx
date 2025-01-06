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

  constexpr double scale_factor = scale_factor_6y;

  BinnedInteraction bint{Ebins, costheta_bins, scale_factor, 40, 40, 1};
  auto cdata = bint.GenerateData(); // data for NH

  BinnedInteraction bint_1{Ebins, costheta_bins, scale_factor, 40, 40, 1};
  bint_1.flip_hierarchy();
  bint_1.UpdatePrediction();
  auto cdata_IH = bint_1.GenerateData();

  // std::println(std::cout, "Mock chi2: {}",
  //              TH2D_chi2(cdata.hist_numu, cdata.hist_numubar));

  auto do_fit_and_plot = [&](double dm32_init,
                             const pull_toggle &toggles = all_on) {
    auto &data_to_fit = dm32_init > 0 ? cdata_IH : cdata;
    MinuitFitter fitter_chi2(bint, data_to_fit, toggles);
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

    double dm32_min = dm32_init > 0 ? 0 : -1;
    double dm32_max = dm32_init > 0 ? 1 : 0;

    auto get_random = [&]() { return 1.; };

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
    auto pre_fit = bint.GenerateData();
    std::println("chi2 pre-fit: {:.4f}",
                 -2 * bint.GetLogLikelihoodAgainstData(data_to_fit));

    auto result = ROOT::Minuit2::MnMigrad{fitter_chi2, param}();

    if (!result.HasValidParameters()) {
      std::println("!!! Warning: HasValidParameters -> false");
    }
    if (!result.IsValid()) {
      std::println("!!! Warning: IsValid -> false");
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

    auto llh_to_data = bint.GetLogLikelihoodAgainstData(data_to_fit);
    auto pull = bint.GetLogLikelihood();

    std::println("chi2 {:.4e}, data: {:.4e}, pull: {:.4e}",
                 -2 * (llh_to_data + pull), -2 * llh_to_data, -2 * pull);

    std::println("{:*^25}\n\n", "finished");
    std::flush(std::cout);
  };

  do_fit_and_plot(-4.91e-3, all_on);
  do_fit_and_plot(4.91e-3, all_on);
  do_fit_and_plot(-4.91e-3, all_off);
  do_fit_and_plot(4.91e-3, all_off);
  for (int i = 0; i < 6; ++i) {
    pull_toggle one_off = all_on;
    one_off[i] = false;
    do_fit_and_plot(-4.91e-3, one_off);
    do_fit_and_plot(4.91e-3, one_off);
    pull_toggle one_on = all_off;
    one_on[i] = true;
    do_fit_and_plot(-4.91e-3, one_on);
    do_fit_and_plot(4.91e-3, one_on);
  }

  return 0;
}
