#include "BinnedInteraction.h"
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
#include <TPad.h>
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
  MinuitFitter(BinnedInteraction &binned_interaction_, SimpleDataHist &data_)
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
    std::println("llh: {}", llh);
    return -2. * llh;
  }

  double Up() const override { return 1.; }

private:
  mutable BinnedInteraction binned_interaction;
  SimpleDataHist data;
};

int main(int argc, char **agrv) {
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat("4.1f");
  TH1::AddDirectory(false);
  auto costheta_bins = linspace(-1., 1., 481);

  auto Ebins = logspace(0.1, 20., 401);

  constexpr double scale_factor = scale_factor_6y;

  BinnedInteraction bint{Ebins, costheta_bins, scale_factor, 40, 40, 1};
  auto cdata = bint.GenerateData(); // data for NH

  MinuitFitter fitter_chi2(bint, cdata);

  ROOT::Minuit2::MnUserParameters param{};
  param.Add("#Delta M_{32}^{2}", -2.455e-3, 0.001e-3, -1, 1);
  param.Add("#Delta M_{21}^{2}", 7.53e-5, 0.01e-5, 0, 1);
  param.Add("sin^{2}#theta_{23}", 0.558, 0.001, 0, 1);
  param.Add("sin^{2}#theta_{13}", 2.19e-2, 0.01e-2, 0, 1);
  param.Add("sin^{2}#theta_{12}", 0.307, 0.001, 0, 1);
  param.Add("#delta_{CP}", 1.19 * M_PI, 0.01, 0, 2 * M_PI);

  auto result = ROOT::Minuit2::MnMigrad{fitter_chi2, param}();

  const auto &final_params = result.UserParameters();
  for (size_t i = 0; i < final_params.Params().size(); ++i) {
    std::println("{}: {}", final_params.GetName(i), final_params.Value(i));
  }

  auto minimal_chi2 = result.Fval();

  std::println("Fval: {}", minimal_chi2);

  return 0;
}
