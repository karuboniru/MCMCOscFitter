#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include <RtypesCore.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TMath.h>
#include <TRandom.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <ranges>

constexpr size_t nbins = 64;

const auto variable_list =
    std::to_array<std::string>({"DM2", "Dm2", "T12", "T13", "T23", "DCP"});
constexpr size_t nvars = variable_list.size();

int locate(int x, int y) { return (x * nvars) + y + 1; }

void do_plot(ROOT::RDF::RNode data_in, const std::string &name) {
  auto canvas = std::make_unique<TCanvas>("c1", "c1", 800 * nvars, 800 * nvars);
  canvas->Divide(nvars, nvars);

  auto minmax = variable_list |
                std::views::transform([&data_in](const auto &var) {
                  auto min = data_in.Min(var);
                  auto max = data_in.Max(var);
                  return std::pair{min, max};
                }) |
                std::ranges::to<std::vector>();

  auto hist1ds = std::views::zip(minmax, variable_list) |
                 std::views::enumerate |
                 std::views::transform([&](const auto &id_tup) {
                   auto &&[id, tup] = id_tup;
                   auto &&[minmax, var] = tup;
                   auto [min, max] = minmax;
                   auto hist = data_in.Histo1D({var.c_str(), var.c_str(), nbins,
                                                min.GetValue(), max.GetValue()},
                                               var);
                   return [&, hist1 = std::move(hist), id]() mutable {
                     canvas->cd(locate(id, id));
                     hist1->Draw();
                     auto separate_canvas = std::make_unique<TCanvas>(
                         std::format("c1_{}", var).c_str(),
                         std::format("c1_{}", var).c_str(), 800, 800);
                     separate_canvas->cd();
                     hist1->Draw();
                     separate_canvas->SaveAs((name + var + ".pdf").c_str());
                     separate_canvas->SaveAs((name + var + ".eps").c_str());
                   };
                 }) |
                 std::ranges::to<std::vector>();

  auto &&coupled_view =
      std::views::zip(minmax, variable_list) | std::views::enumerate;

  auto plot_2d_action =
      std::views::cartesian_product(coupled_view, coupled_view) |
      std::views::filter([](auto &&v) -> bool {
        auto &&[x, y] = v;
        auto id_x = std::get<0>(x);
        auto id_y = std::get<0>(y);
        return id_x > id_y;
      }) |
      std::views::transform([&data_in, &canvas](auto &&v) {
        auto &&[x, y] = v;
        auto &&[id_x, tup_x] = x;
        auto &&[id_y, tup_y] = y;
        auto &&[minmax_x, var_x] = tup_x;
        auto &&[min_x, max_x] = minmax_x;
        auto &&[minmax_y, var_y] = tup_y;
        auto &&[min_y, max_y] = minmax_y;
        auto name = std::format("{}_vs_{}", var_x, var_y);
        auto hist2d = data_in.Histo2D(
            {name.c_str(), name.c_str(), nbins, min_y.GetValue(),
             max_y.GetValue(), nbins, min_x.GetValue(), max_x.GetValue()},
            var_y, var_x);
        return [&canvas, id_x, id_y, this_hist = std::move(hist2d)]() mutable {
          canvas->cd(locate(id_x, id_y));
          this_hist->Draw("colz");
        };
      }) |
      std::ranges::to<std::vector>();

  std::ranges::for_each(hist1ds, [](auto &f) { f(); });
  std::ranges::for_each(plot_2d_action, [](auto &f) { f(); });

  canvas->SaveAs((name + ".pdf").c_str());
  canvas->SaveAs((name + ".eps").c_str());
  canvas->SaveAs((name + ".svg").c_str());
}

int main(int argc, char **argv) {
  ROOT::EnableImplicitMT();
  TH1::AddDirectory(false);

  auto data_in = ROOT::RDataFrame{"tree", "testfit.root"}.Filter(
      [](size_t count) { return count > 10000; }, {"count"});

  do_plot(data_in.Filter([](double DM2) { return DM2 > 0; }, {"DM2"}), "NH");
  do_plot(data_in.Filter([](double DM2) { return DM2 < 0; }, {"DM2"}), "IH");
  do_plot(data_in.Redefine("DM2", [](double DM2) { return std::abs(DM2); },
                           {"DM2"}),
          "ALL");

  return 0;
}
