#include "OscillationParameters.h"
#include "ParBinnedInterface.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "timer.hpp"

#include <TMath.h>
#include <TRandom.h>
#include <array>
#include <cmath>
#include <fstream>
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
void print_parameters(const param &p) {
  std::println(
      "DM2: {:.2e} T23: {:.2e} T13: {:.2e} Dm2: {:.2e} T12: {:.2e} DCP: {:.2e}",
      p.DM2, p.T23, p.T13, p.Dm2, p.T12, p.DCP);
}

} // namespace

SimpleDataHist rebin_new_method(const SimpleDataHist &from,
                                std::vector<double> ebin_edges,
                                std::vector<double> costh_bin_edges) {
  SimpleDataHist ret{};

  auto TH2D_rebin_new = [&](const TH2D &from_hist) {
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

  ParBinnedInterface bint{Ebins, costheta_bins, scale_factor, e_rebin_frac,
                          costh_rebin_frac};
  auto cdata = bint.GenerateData();
  auto rebin_wing = [&](const SimpleDataHist &d) {
    return rebin_new_method(d, e_bin_wing, costh_bin_wing);
  };
  auto cdata_NH_rebinned = rebin_wing(cdata);
  auto param_NH = bint.get_param();
  // print_parameters(param_NH);
  std::println("{}", param_NH);
  bint.flip_hierarchy();
  auto cdata_IH = bint.GenerateData();
  auto cdata_IH_rebinned = rebin_wing(cdata_IH);
  auto param_IH = bint.get_param();
  // print_parameters(param_IH);

  double chi2_pred_IH = chi2_data(cdata_NH_rebinned, cdata_IH_rebinned);
  double chi2_pred_NH = chi2_data(cdata_IH_rebinned, cdata_NH_rebinned);

  std::println("NH PRED: {:.2f}\nIH: PRED {:.2f}", chi2_pred_NH, chi2_pred_IH);

  auto create_xcheck_file = [&](double param::*param_to_vary, double min,
                                double max, size_t steps,
                                const std::string &prefix) {
    std::fstream NHtrue(prefix + "_NHtrue.txt",
                        std::ios::trunc | std::ios::out);
    std::fstream IHtrue(prefix + "_IHtrue.txt",
                        std::ios::trunc | std::ios::out);
    std::println(NHtrue, "#Asimov from: {}", param_NH);
    std::println(IHtrue, "#Asimov from: {}", param_IH);

    std::println(NHtrue, "var\t\t'chi2 (data: Asimov NH) pred: (varied IH)'");
    std::println(IHtrue, "var\t'chi2 (data: Asimov IH) pred: (varied NH)'");

    for (size_t i{}; i <= steps; i++) {
      double var = min + ((max - min) / steps * i);
      auto this_param_NH = param_NH;
      this_param_NH.*param_to_vary = var;
      bint.set_param(this_param_NH);
      auto pred_NH = bint.GenerateData();
      auto pred_rebinned_NH = rebin_wing(pred_NH);
      double chi2_IHTRUE = chi2_data(cdata_IH_rebinned, pred_rebinned_NH);
      auto this_param_IH = param_IH;
      this_param_IH.*param_to_vary = var;
      this_param_IH.DM2 = -std::abs(this_param_IH.DM2);
      bint.set_param(this_param_IH);
      auto pred_IH = bint.GenerateData();
      auto pred_rebinned_IH = rebin_wing(pred_IH);
      double chi2_NHTRUE = chi2_data(cdata_NH_rebinned, pred_rebinned_IH);
      std::println(NHtrue, "{}\t\t{}", var, chi2_NHTRUE);
      std::println(IHtrue, "{}\t\t{}", var, chi2_IHTRUE);
    }
  };

  create_xcheck_file(&param::DCP, 0, 2 * M_PI, 100, "dcp_100");
  create_xcheck_file(&param::DM2, 2e-3, 3e-3, 100, "dm32_100");
  create_xcheck_file(&param::T23, 0.3, 0.7, 100, "t23_100");

  return 0;
}