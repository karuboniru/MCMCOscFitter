#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include <BinnedInteraction.h>

#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnContours.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnUserParameters.h>

#include <TCanvas.h>
#include <TGraph.h>

#include <algorithm>
#include <cmath>
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
    return -2. * llh;
  }

  double Up() const override { return 1.; }

private:
  mutable BinnedInteraction binned_interaction;
  SimpleDataHist data;
};

int main(int argc, char **agrv) {
  auto costheta_bins = linspace(-1., 1., 481);

  auto Ebins = logspace(0.1, 20., 401);

  constexpr double scale_factor =
      (2e10 / (12 + H_to_C) * 6.02214076e23) * // number of target C12
      ((6 * 365) * 24 * 3600) /                // seconds in a year
      1e42; // unit conversion from 1e-38 cm^2 to 1e-42 m^2

  BinnedInteraction bint{Ebins, costheta_bins, scale_factor, 40, 40, 1};
  auto cdata = bint.GenerateData(); // data for NH

  std::println("event count for NH: {:.2f}", cdata.hist_numu.GetSum());

  MinuitFitter fitter_IH(bint, cdata);
  ROOT::Minuit2::MnUserParameters param{};
  param.Add("#Delta M_{32}^{2}", -2.529e-3, 0.029e-3, -1, 0);
  // param.Add("#Delta M_{32}^{2}", 2.529e-3, 0.029e-3, 0, 1);
  param.Add("#Delta M_{21}^{2}", 7.53e-5, 0.18e-5, 0, 1);
  param.Add("sin^{2}#theta_{23}", 0.553, 0.022, 0, 1);
  param.Add("sin^{2}#theta_{13}", 2.19e-2, 0.07e-2, 0, 1);
  param.Add("sin^{2}#theta_{12}", 0.307, 0.013, 0, 1);
  param.Add("#delta_{CP}", 1.36 * M_PI, 0.2);

  auto result = ROOT::Minuit2::MnMigrad{fitter_IH, param}();

  const auto &final_params = result.UserParameters();
  for (size_t i = 0; i < final_params.Params().size(); ++i) {
    std::println("{}: {}", final_params.GetName(i), final_params.Value(i));
  }

  auto minimal_chi2 = result.Fval();

  std::println("Fval: {}", minimal_chi2);

  ROOT::Minuit2::MnContours contours(fitter_IH, result);
  for (size_t i = 0; i < final_params.Params().size(); ++i) {
    for (size_t j = i + 1; j < final_params.Params().size(); ++j) {
      const size_t npoints = 15;
      auto contour = contours(i, j, npoints);
      std::println("Contour between {} and {}:", final_params.GetName(i),
                   final_params.GetName(j));
      std::array<double, npoints + 1> x{}, y{};
      std::ranges::for_each(std::views::zip(contour, x, y),
                            [](auto &&point_x_y) {
                              auto &[point, x, y] = point_x_y;
                              x = point.first;
                              y = point.second;
                            });
      x[npoints] = x[0];
      y[npoints] = y[0];
      TCanvas canvas;
      auto left_margin = canvas.GetLeftMargin();
      canvas.SetLeftMargin(left_margin * 1.5);

      auto right_margin = canvas.GetRightMargin();
      canvas.SetRightMargin(right_margin * 0.5);

      TGraph graph(npoints + 1, x.data(), y.data());
      graph.SetTitle("");
      graph.GetXaxis()->SetTitle(final_params.GetName(i).c_str());
      graph.GetYaxis()->SetTitle(final_params.GetName(j).c_str());

      graph.Draw("AC");
      canvas.SaveAs(std::format("contour_{}_{}.pdf", final_params.GetName(i),
                                final_params.GetName(j))
                        .c_str());
    }
  }

  return 0;
}