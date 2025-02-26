#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include <TFile.h>
#include <TH2D.h>
#include <array>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <ranges>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::println(std::cerr, "Usage: {} data_file pred_file", argv[0]);
    return 1;
  }
  SimpleDataHist data;
  data.LoadFrom(argv[1]);
  std::ofstream out(argv[2]);
  auto &numu = data.hist_numu;
  auto &nue = data.hist_nue;
  auto &numubar = data.hist_numubar;
  auto &nuebar = data.hist_nuebar;
  std::println(out, "## columns are: energyBin_lowEdge cosThetaBin_lowEdge   "
                    "numu  numubar  nue  nuebar nc   ###");
  std::println(out, "####------------------------------------------------------"
                    "----------####");
  for (int e_index = 1; e_index <= numu.GetNbinsX(); ++e_index) {
    for (int c_index = 1; c_index <= numu.GetNbinsY(); ++c_index) {
      std::println(out, "{} {} {} {} {} {} {}",
                   numu.GetXaxis()->GetBinLowEdge(e_index),
                   numu.GetYaxis()->GetBinLowEdge(c_index),
                   numu.GetBinContent(e_index, c_index),
                   numubar.GetBinContent(e_index, c_index),
                   nue.GetBinContent(e_index, c_index),
                   nuebar.GetBinContent(e_index, c_index), 0);
    }
  }
  std::println(out, "## ----------------------------------------------###");
  std::println(out, "##total event after all {} {} {} {}",

               numu.Integral(), numubar.Integral(), nue.Integral(),
               nuebar.Integral());
  return 0;
}
