#include "OscillationParameters.h"
#include "ParBinnedInterface.h"
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
#include <array>
#include <cmath>
#include <filesystem>
#include <omp.h>
#include <print>
#include <ranges>

using fit_input_t = std::tuple<double param::* /*PARAM to vary*/,
                               double /*min*/, double /*max*/>;
auto inputs =
    std::to_array<fit_input_t>({std::make_tuple(&param::DM2, 0.0, 4e-3),
                                std::make_tuple(&param::T23, 0, 0.5),
                                std::make_tuple(&param::DCP, 0, 2 * M_PI)});
std::string to_name(double param::*ptr) {
  if (ptr == &param::DM2) {
    return "DM2";
  } else if (ptr == &param::T23) {
    return "T23";
  } else if (ptr == &param::DCP) {
    return "DCP";
  }
  throw std::runtime_error("Unknown parameter");
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
            (bin_pred - bin_data) + bin_data * TMath::Log(bin_data / bin_pred);
      else
        chi2 += bin_pred;
    }
  }
  return 2 * chi2;
}

double chi2_data(const SimpleDataHist &data, const SimpleDataHist &pred) {
  double chi2{};
  chi2 += TH2D_chi2(data.hist_numu, pred.hist_numu);
  chi2 += TH2D_chi2(data.hist_numubar, pred.hist_numubar);
  chi2 += TH2D_chi2(data.hist_nue, pred.hist_nue);
  chi2 += TH2D_chi2(data.hist_nuebar, pred.hist_nuebar);
  return chi2;
}
} // namespace

class MinuitFitter final : public ROOT::Minuit2::FCNBase {
public:
  MinuitFitter(ParBinnedInterface &binned_interaction_, SimpleDataHist &data_,
               param param_, std::vector<double> ebin_edges_,
               std::vector<double> costh_bin_edges_)
      : binned_interaction(binned_interaction_), data(data_), par(param_),
        ebin_edges(ebin_edges_), costh_bin_edges(costh_bin_edges_) {}

  double operator()(const std::vector<double> &params) const override {
    // auto t_param = params[0];
    // if (param_ptr == &param::DM2) {
    //   // always fit with wrong sign
    //   if (par.DM2 < 0) {
    //     t_param = -t_param;
    //   }
    // }
    // par.*param_ptr = t_param;
    auto new_param = par;
    // new_param.*param_ptr = t_param;
    for (auto [addr, var] :
         std::views::zip(inputs | std::views::keys, params)) {
      auto thisvar = var;
      if (addr == &param::DM2) {
        if (par.DM2 < 0) {
          thisvar = -thisvar;
        }
      }
      new_param.*addr = thisvar;
    }
    binned_interaction.set_param(new_param);
    auto pred_rebinned = binned_interaction.GenerateData();
    // auto pred_rebinned = rebin_new_method(pred, ebin_edges, costh_bin_edges);
    auto chi2 = chi2_data(data, pred_rebinned);
    return chi2;
  }

  double Up() const override { return 1.; }

private:
  mutable ParBinnedInterface binned_interaction;
  SimpleDataHist data;
  param par;
  std::vector<double> ebin_edges, costh_bin_edges;
};

auto get_fit_par() {
  auto get_random = [&]() { return gRandom->Uniform(0.1, 1.9); };

  ROOT::Minuit2::MnUserParameters fitparNH{};
  for (auto &&[var, min, max] : inputs) {
    fitparNH.Add(to_name(var), (min + max) / 2 * get_random(),
                 (max - min) / 500, min, max);
  }
  return fitparNH;
}

int main(int argc, char **agrv) {
  // gSystem->Lo
  gErrorIgnoreLevel = kWarning;
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat("4.1f");
  TH1::AddDirectory(false);
  auto costheta_bins = linspace(-1., 1., 401);

  auto Ebins = logspace(0.1, 20., 401);

  constexpr double scale_factor = scale_factor_6y;

  auto e_bin_wing =
      std::vector<double>{0.1, 0.6, 0.8, 1.0, 1.35, 1.75, 2.2, 3.0, 4.6, 20.0};
  auto costh_bin_wing = linspace(-1., 1., 10 + 1);

  ParBinnedInterface bint{Ebins, costheta_bins, scale_factor, e_bin_wing,
                          costh_bin_wing};
  auto cdata_NH_rebinned = bint.GenerateData(); // data for NH
  // auto cdata_NH_rebinned =
  //     rebin_new_method(cdata_NH, e_bin_wing, costh_bin_wing);
  auto par_NH = bint.get_param();

  // ParBinnedInterface bint_1{Ebins, costheta_bins, scale_factor, 40, 40, 1};
  bint.flip_hierarchy();
  auto cdata_IH_rebinned = bint.GenerateData(); // data for IH
  // auto cdata_IH_rebinned =
  //     rebin_new_method(cdata_IH, e_bin_wing, costh_bin_wing);
  auto par_IH = bint.get_param();

  MinuitFitter fitter_NH(bint, cdata_NH_rebinned, par_IH, e_bin_wing,
                         costh_bin_wing);
  MinuitFitter fitter_IH(bint, cdata_IH_rebinned, par_NH, e_bin_wing,
                         costh_bin_wing);

  double bestChi2_NH = std::numeric_limits<double>::infinity();
  double bestChi2_IH = std::numeric_limits<double>::infinity();
  std::vector<double> bestParam_NH, bestParam_IH;

  for (int i = 0; i < 32; ++i) {
    auto fitparNH = get_fit_par();
    auto resultNH = ROOT::Minuit2::MnMigrad{fitter_NH, fitparNH}();
    if (resultNH.Fval() < bestChi2_NH) {
      bestChi2_NH = resultNH.Fval();
      bestParam_NH = resultNH.UserParameters().Params();
    }

    auto fitparIH = get_fit_par();
    auto resultIH = ROOT::Minuit2::MnMigrad{fitter_IH, fitparIH}();
    if (resultIH.Fval() < bestChi2_IH) {
      bestChi2_IH = resultIH.Fval();
      bestParam_IH = resultIH.UserParameters().Params();
    }
  }

  std::cout << "Best Chi2 NH: " << bestChi2_NH << "\n";
  std::cout << "Best Param NH: ";
  for (auto v : bestParam_NH) {
    std::cout << v << " ";
  }
  std::cout << "\n";
  std::cout << "Best Chi2 IH: " << bestChi2_IH << "\n";
  std::cout << "Best Param IH: ";
  for (auto v : bestParam_IH) {
    std::cout << v << " ";
  }
  std::cout << "\n";

  return 0;
}
