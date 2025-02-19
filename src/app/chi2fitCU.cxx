#include "OscillationParameters.h"
#include "ParBinned.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
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
            (bin_pred - bin_data) + bin_data * TMath::Log(bin_data / bin_pred);
      else
        chi2 += bin_pred;
    }
  }
  return 2 * chi2;
}
} // namespace

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
  ret.hist_numu = lhs.hist_numu / rhs.hist_numu;
  ret.hist_numubar = lhs.hist_numubar / rhs.hist_numubar;
  ret.hist_nue = lhs.hist_nue / rhs.hist_nue;
  ret.hist_nuebar = lhs.hist_nuebar / rhs.hist_nuebar;
  return ret;
}

double chi2_possion(double bin_data, double bin_pred) {
  return ((bin_pred - bin_data) + bin_data * log(bin_data / bin_pred)) * 2;
}

TH2D get_chi2_hist(const TH2D &data, const TH2D &pred) {
  TH2D ret = data;
  ret.Clear();

  for (int i = 1; i <= data.GetNbinsX(); ++i) {
    for (int j = 1; j <= data.GetNbinsY(); ++j) {
      auto data_val = data.GetBinContent(i, j);
      auto pred_val = pred.GetBinContent(i, j);
      if (data_val != 0) {
        ret.SetBinContent(i, j, chi2_possion(data_val, pred_val));
      } else {
        ret.SetBinContent(i, j, pred_val);
      }
    }
  }
  return ret;
}

SimpleDataHist get_chi2_data(const SimpleDataHist &data,
                             const SimpleDataHist &pred) {
  SimpleDataHist ret;
  ret.hist_numu = get_chi2_hist(data.hist_numu, pred.hist_numu);
  ret.hist_numubar = get_chi2_hist(data.hist_numubar, pred.hist_numubar);
  ret.hist_nue = get_chi2_hist(data.hist_nue, pred.hist_nue);
  ret.hist_nuebar = get_chi2_hist(data.hist_nuebar, pred.hist_nuebar);
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
  reset_style(data.hist_numu, "#nu_{#mu}");
  data.hist_numu.Draw("COLZ TEXT");
  c1.cd(2)->SetLogx();
  reset_style(data.hist_numubar, "#bar{#nu}_{#mu}");
  data.hist_numubar.Draw("COLZ TEXT");
  c1.cd(3)->SetLogx();
  reset_style(data.hist_nue, "#nu_{e}");
  data.hist_nue.Draw("COLZ TEXT");
  c1.cd(4)->SetLogx();
  reset_style(data.hist_nuebar, "#bar{#nu}_{e}");
  data.hist_nuebar.Draw("COLZ TEXT");
  c1.SaveAs(filename.c_str());
}

int main(int argc, char **agrv) {
  // gSystem->Lo
  gErrorIgnoreLevel = kWarning;
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat("4.1f");
  TH1::AddDirectory(false);
  auto costheta_bins = linspace(-1., 1., 481);

  auto Ebins = logspace(0.1, 20., 401);

  constexpr double scale_factor = scale_factor_6y;

  ParBinnedInterface bint{Ebins, costheta_bins, scale_factor, 40, 40, 1};
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
                                    return do_fit_and_plot(-4.91e-3, one_off,
                                                           6);
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
