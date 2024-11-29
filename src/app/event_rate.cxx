#include "BinnedInteraction.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "timer.hpp"

#include <TMath.h>
#include <TRandom.h>
#include <cmath>
#include <print>

int main(int argc, char **argv) {
  auto costheta_bins = linspace(-1., 1., 401);

  auto Ebins = logspace(0.1, 20., 401);

  constexpr double scale_factor =
      (2e10 / (12 + H_to_C) * 6.022e23) * (6 * 365 * 24 * 3600) / 1e42;

  BinnedInteraction bint{Ebins, costheta_bins, scale_factor, 1, 1};
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

  std::cout << "\nNon-Oscillated Event Rates:" << std::endl;
  std::cout << "numu: " << no_osc_numu << std::endl;
  std::cout << "numu_bar: " << no_osc_numu_bar << std::endl;
  std::cout << "nue: " << no_osc_nue << std::endl;
  std::cout << "nue_bar: " << no_osc_nue_bar << std::endl;

  std::cout << "\nOscillated Event Rates:" << std::endl;
  std::cout << "numu: " << numu << std::endl;
  std::cout << "numu_bar: " << numu_bar << std::endl;
  std::cout << "nue: " << nue << std::endl;
  std::cout << "nue_bar: " << nue_bar << std::endl;

  bint.flip_hierarchy();
  auto cdata_IH = bint.GenerateData();

  auto numu_IH = cdata_IH.hist_numu.Integral();
  auto numu_bar_IH = cdata_IH.hist_numubar.Integral();
  auto nue_IH = cdata_IH.hist_nue.Integral();
  auto nue_bar_IH = cdata_IH.hist_nuebar.Integral();

  std::cout << "\nInverted Hierarchy Event Rates:" << std::endl;
  std::cout << "numu: " << numu_IH << std::endl;
  std::cout << "numu_bar: " << numu_bar_IH << std::endl;
  std::cout << "nue: " << nue_IH << std::endl;
  std::cout << "nue_bar: " << nue_bar_IH << std::endl;

  return 0;
}