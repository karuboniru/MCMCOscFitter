#include "BinnedInteraction.h"
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
    std::println("|{:^10}|{:^8.1f}|{:^8.0f}|{:^6.1f}‰|", name, res, ref,
                 1000. * (res - ref) / ref);
  }
  std::println("{}\n", hline);
}

int main(int argc, char **argv) {
  auto costheta_bins = linspace(-1., 1., 401);

  auto Ebins = logspace(0.1, 20., 401);

  constexpr double scale_factor = scale_factor_6y;

  BinnedInteraction bint{Ebins, costheta_bins, scale_factor, 40, 40};
  auto cdata = bint.GenerateData();
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
         {7012.66, 2600.31, 3622.82, 1172.16});

  report("Normal Hierarchy Event Rates", {numu, numu_bar, nue, nue_bar},
         {4820.17, 1805.62, 3739.65, 1153.6});
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
         {4829.29, 1791.67, 3686.88, 1171.33});

  bint.Save_prob_hist("IH.root");

  return 0;
}