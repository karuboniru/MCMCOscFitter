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

SimpleDataHist rebin_new_method(SimpleDataHist &from,
                                std::vector<double> ebin_edges,
                                std::vector<double> costh_bin_edges) {
  SimpleDataHist ret{};

  auto TH2D_rebin_new = [&](TH2D &from_hist) {
    TH2D new_hist(from_hist.GetName(), from_hist.GetName(),
                  ebin_edges.size() - 1, ebin_edges.data(),
                  costh_bin_edges.size() - 1, costh_bin_edges.data());
    for (int i = 1; i <= from_hist.GetNbinsX(); ++i) {
      for (int j = 1; j <= from_hist.GetNbinsY(); ++j) {
        std::map<int, double> bin_map_x,
            bin_map_y; // bin id on x/y, fraction to assign
        auto from_bin_content = from_hist.GetBinContent(i, j);

        auto from_bin_lower_e = from_hist.GetXaxis()->GetBinLowEdge(i);
        auto from_bin_upper_e = from_hist.GetXaxis()->GetBinUpEdge(i);

        auto new_bin_lower_x = new_hist.GetXaxis()->FindBin(from_bin_lower_e);
        auto new_bin_upper_x = new_hist.GetXaxis()->FindBin(from_bin_upper_e);
        if (new_bin_lower_x == new_bin_upper_x) { // in a single bin
          bin_map_x[new_bin_lower_x] = 1.0;
        } else {
          auto x_div = new_hist.GetXaxis()->GetBinLowEdge(new_bin_upper_x);
          bin_map_x[new_bin_lower_x] = (x_div - from_bin_lower_e) /
                                       (from_bin_upper_e - from_bin_lower_e);
          bin_map_x[new_bin_upper_x] = (from_bin_upper_e - x_div) /
                                       (from_bin_upper_e - from_bin_lower_e);
        }

        auto from_bin_lower_costh = from_hist.GetYaxis()->GetBinLowEdge(j);
        auto from_bin_upper_costh = from_hist.GetYaxis()->GetBinUpEdge(j);

        auto new_bin_lower_y =
            new_hist.GetYaxis()->FindBin(from_bin_lower_costh);
        auto new_bin_upper_y =
            new_hist.GetYaxis()->FindBin(from_bin_upper_costh);
        if (new_bin_lower_y == new_bin_upper_y) { // in a single bin
          bin_map_y[new_bin_lower_y] += 1.0;
        } else {
          auto y_div = new_hist.GetYaxis()->GetBinLowEdge(new_bin_upper_y);
          bin_map_y[new_bin_lower_y] =
              (y_div - from_bin_lower_costh) /
              (from_bin_upper_costh - from_bin_lower_costh);
          bin_map_y[new_bin_upper_y] =
              (from_bin_upper_costh - y_div) /
              (from_bin_upper_costh - from_bin_lower_costh);
        }

        for (auto &&[bin_id, fraction] : bin_map_x) {
          for (auto &&[bin_id_y, fraction_y] : bin_map_y) {
            new_hist.AddBinContent(bin_id, bin_id_y,
                                   from_bin_content * fraction * fraction_y);
          }
        }
      }
    }
    return new_hist;
  };
  ret.hist_numu = TH2D_rebin_new(from.hist_numu);
  ret.hist_nue = TH2D_rebin_new(from.hist_nue);
  ret.hist_numubar = TH2D_rebin_new(from.hist_numubar);
  ret.hist_nuebar = TH2D_rebin_new(from.hist_nuebar);
  return ret;
}

int main(int argc, char **argv) {
  TH1::AddDirectory(false);
  constexpr size_t e_rebin_frac = 1;
  constexpr size_t e_bin_count = 400;
  constexpr size_t costh_rebin_frac = 1;
  constexpr size_t costh_bin_count = 400;
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
  auto cdata_rebinned = rebin_new_method(cdata, e_bin_wing, costh_bin_wing);
  cdata_rebinned.SaveAs("Event_rate_NH.root");
  auto cdata_noOsc = bint.GenerateData_NoOsc();
  std::println("nc event count: {:.3f}",
               cdata_noOsc.hist_nc->GetSumOfWeights());
  auto cdata_noOsc_rebinned =
      rebin_new_method(cdata_noOsc, e_bin_wing, costh_bin_wing);
  cdata_noOsc_rebinned.SaveAs("No_Osc.root");
  auto cdata_IH = bint.GenerateData();
  auto cdata_IH_rebinned =
      rebin_new_method(cdata_IH, e_bin_wing, costh_bin_wing);
  cdata_IH_rebinned.SaveAs("Event_rate_IH.root");

  return 0;
}