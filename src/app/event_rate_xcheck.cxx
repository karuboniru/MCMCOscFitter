#include "ParBinnedInterface.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "timer.hpp"

#include <TMath.h>
#include <TRandom.h>
#include <array>
#include <cmath>
#include <map>
#include <print>
#include <ranges>
#include <string>
#include <string_view>

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

int main(int argc, char **argv) {
  TH1::AddDirectory(false);
  constexpr size_t e_rebin_frac = 1;
  constexpr size_t e_bin_count = 400;
  constexpr size_t costh_rebin_frac = 10;
  constexpr size_t costh_bin_count = 40;
  auto e_bin_wing =
      std::vector<double>{0.1, 0.6, 0.8, 1.0, 1.35, 1.75, 2.2, 3.0, 4.6, 20.0};
  auto costh_bin_wing = linspace(-1., 1., 10 + 1);

  //   auto Ebins = divide_bins<double>(e_bin_wing, e_rebin_frac);
  auto Ebins = logspace(0.1, 20., (e_bin_count * e_rebin_frac) + 1);
  auto costheta_bins =
      linspace(-1., 1., (costh_rebin_frac * costh_bin_count) + 1);

  constexpr double scale_factor = scale_factor_6y;

  ParBinnedInterface bint{Ebins, costheta_bins, scale_factor, e_bin_wing,
                          costh_bin_wing};
  auto cdata_NH_rebinned = bint.GenerateData();
  // auto cdata_NH_rebinned = rebin_new_method(cdata, e_bin_wing,
  // costh_bin_wing);
  cdata_NH_rebinned.SaveAs("Event_rate_NH.root");
  auto cdata_noOsc_rebinned = bint.GenerateData_NoOsc();
  std::println("nc event count: {:.3f}",
               cdata_noOsc_rebinned.hist_nc->GetSumOfWeights());
  // auto cdata_noOsc_rebinned =
  //     rebin_new_method(cdata_noOsc, e_bin_wing, costh_bin_wing);
  cdata_noOsc_rebinned.SaveAs("No_Osc.root");
  bint.flip_hierarchy();
  auto cdata_IH_rebinned = bint.GenerateData();
  // auto cdata_IH_rebinned =
  //     rebin_new_method(cdata_IH, e_bin_wing, costh_bin_wing);
  cdata_IH_rebinned.SaveAs("Event_rate_IH.root");

  double chi2_pred_IH = chi2_data(cdata_NH_rebinned, cdata_IH_rebinned);
  double chi2_pred_NH = chi2_data(cdata_IH_rebinned, cdata_NH_rebinned);

  std::println("NH PRED: {:.2f}\nIH: PRED {:.2f}", chi2_pred_NH, chi2_pred_IH);

  return 0;
}