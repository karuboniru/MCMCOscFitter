#include "SimpleDataHist.h"
#include <print>

namespace {
double TH2D_chi2(const TH2D &data, const TH2D &pred) {
  auto binsx = data.GetNbinsX();
  auto binsy = data.GetNbinsY();
  double chi2{};
#pragma omp parallel for reduction(+ : chi2) collapse(2)
  for (int x = 1; x <= binsx; x++) {
    for (int y = 1; y <= binsy; y++) {
      auto bin_data = data.GetBinContent(x, y);
      auto bin_pred = pred.GetBinContent(x, y);
      if (bin_data != 0) [[likely]]
        chi2 +=
            (bin_pred - bin_data) + bin_data * std::log(bin_data / bin_pred);
      else
        chi2 += bin_pred;
    }
  }
  return 2 * chi2;
}
} // namespace

int main(int argc, char **argv) {
  SimpleDataHist data, pred;
  data.LoadFrom(argv[1]);
  pred.LoadFrom(argv[2]);
  auto chi2 = TH2D_chi2(data.hist_numu, pred.hist_numu) +
              TH2D_chi2(data.hist_nue, pred.hist_nue) +
              TH2D_chi2(data.hist_numubar, pred.hist_numubar) +
              TH2D_chi2(data.hist_nuebar, pred.hist_nuebar);
  std::println("chi2: {}", chi2);
}