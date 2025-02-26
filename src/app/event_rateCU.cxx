#include "ParBinnedInterface.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "timer.hpp"

#include <TMath.h>
#include <TRandom.h>
#include <array>
#include <cmath>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
void report(std::string_view title, const std::array<double, 4> &result,
            const std::array<double, 4> &ref) {
  const std::string hline(38, '-');
  const auto name =
      std::to_array<std::string_view>({"numu", "numubar", "nue", "nuebar"});
  std::println("{}", hline);
  std::println("|{:^36}|", title);
  std::println("{}", hline);
  std::println("|{:^10}|{:^8}|{:^8}|{:^7}|", "Flavor", "Result", "Ziou",
               "Diff");
  std::println("{}", hline);
  for (auto &&[name, res, ref] : std::ranges::views::zip(name, result, ref)) {
    std::println("|{:^10}|{:^8.3f}|{:^8.3f}|{:^6.2f}%|", name, res, ref,
                 100. * (res - ref) / ref);
  }
  std::println("{}\n", hline);
}

int main(int argc, char **argv) {
  TH1::AddDirectory(false);
  constexpr size_t e_rebin_frac = 40;
  constexpr size_t e_bin_count = 10;
  constexpr size_t costh_rebin_frac = 40;
  constexpr size_t costh_bin_count = 12;
  auto e_bin_wing =
      std::vector<double>{0.1, 0.6, 0.8, 1.0, 1.35, 1.75, 2.2, 3.0, 4.6, 20.0};
  auto costh_bin_wing = linspace(-1., 1., 10 + 1);

  //   auto Ebins = divide_bins<double>(e_bin_wing, e_rebin_frac);
  auto Ebins = logspace(0.1, 20., (e_bin_count * e_rebin_frac) + 1);
  auto costheta_bins =
      linspace(-1., 1., (costh_rebin_frac * costh_bin_count) + 1);

  constexpr double scale_factor = scale_factor_6y;

  ParBinnedInterface bint{Ebins, costheta_bins, scale_factor, e_rebin_frac,
                          costh_rebin_frac};
  auto cdata = bint.GenerateData();
  bint.SaveAs("flux_xsec.root");
  auto cdata_noOsc = bint.GenerateData_NoOsc();

  auto numu = cdata.hist_numu.Integral();
  auto numu_bar = cdata.hist_numubar.Integral();
  auto nue = cdata.hist_nue.Integral();
  auto nue_bar = cdata.hist_nuebar.Integral();

  auto no_osc_numu = cdata_noOsc.hist_numu.Integral();
  auto no_osc_numu_bar = cdata_noOsc.hist_numubar.Integral();
  auto no_osc_nue = cdata_noOsc.hist_nue.Integral();
  auto no_osc_nue_bar = cdata_noOsc.hist_nuebar.Integral();

  report("Non-Oscillated Event Rates",
         {no_osc_numu, no_osc_numu_bar, no_osc_nue, no_osc_nue_bar},
         {7004.60, 2591.95, 3618.78, 1168.26});

  report("Normal Hierarchy Event Rates", {numu, numu_bar, nue, nue_bar},
         {4814.72, 1800.04, 3735.50, 1149.77});
  cdata_noOsc.SaveAs("No_Osc.root");
  cdata.SaveAs("Event_rate_NH.root");
  bint.Save_prob_hist("NH.root");

  bint.flip_hierarchy();
  auto cdata_IH = bint.GenerateData();

  auto numu_IH = cdata_IH.hist_numu.Integral();
  auto numu_bar_IH = cdata_IH.hist_numubar.Integral();
  auto nue_IH = cdata_IH.hist_nue.Integral();
  auto nue_bar_IH = cdata_IH.hist_nuebar.Integral();

  cdata_IH.SaveAs("Event_rate_IH.root");

  report("Inverted Hierarchy Event Rates",
         {numu_IH, numu_bar_IH, nue_IH, nue_bar_IH},
         {4829.87, 1788.41, 3682.56, 1167.63});

  bint.Save_prob_hist("IH.root");

  return 0;
}
