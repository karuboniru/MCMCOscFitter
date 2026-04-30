#include "BinnedInteraction.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "fit_config.h"
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
    std::println("|{:^10}|{:^8.1f}|{:^8.0f}|{:^6.2f}%|", name, res, ref,
                 100. * (res - ref) / ref);
  }
  std::println("{}\n", hline);
}

int main(int argc, char **argv) {
  TH1::AddDirectory(false);
  auto Ebins = logspace(FitConfig::e_min, FitConfig::e_max, FitConfig::n_energy_bins + 1);
  auto costheta_bins = linspace(-1., 1., FitConfig::n_costheta_bins + 1);

  BinnedInteraction bint{Ebins, costheta_bins, FitConfig::scale_factor,
                         FitConfig::E_rebin_factor, FitConfig::costh_rebin_factor};
  auto cdata = bint.GenerateData();
  bint.SaveAs("flux_xsec.root");
  auto cdata_noOsc = bint.GenerateData_NoOsc();

  auto numu = cdata.total_numu();
  auto numu_bar = cdata.total_numubar();
  auto nue = cdata.total_nue();
  auto nue_bar = cdata.total_nuebar();

  auto no_osc_numu = cdata_noOsc.total_numu();
  auto no_osc_numu_bar = cdata_noOsc.total_numubar();
  auto no_osc_nue = cdata_noOsc.total_nue();
  auto no_osc_nue_bar = cdata_noOsc.total_nuebar();

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

  auto numu_IH = cdata_IH.total_numu();
  auto numu_bar_IH = cdata_IH.total_numubar();
  auto nue_IH = cdata_IH.total_nue();
  auto nue_bar_IH = cdata_IH.total_nuebar();

  cdata_IH.SaveAs("Event_rate_IH.root");

  report("Inverted Hierarchy Event Rates",
         {numu_IH, numu_bar_IH, nue_IH, nue_bar_IH},
         {4829.29, 1791.67, 3686.88, 1171.33});

  bint.Save_prob_hist("IH.root");

  return 0;
}