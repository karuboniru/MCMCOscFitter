#pragma once

#include <TH2.h>
#include <cmath>

// Poisson log-likelihood chi2 statistic:
//   2 * sum_bins [ pred - data + data * ln(data/pred) ]   (data > 0)
//   2 * pred                                               (data == 0)
// Returns 0 when pred == data everywhere.
inline double TH2D_chi2(const TH2D &data, const TH2D &pred) {
  const int binsx = data.GetNbinsX();
  const int binsy = data.GetNbinsY();
  double chi2{};
  #pragma omp parallel for reduction(+ : chi2) collapse(2)
  for (int x = 1; x <= binsx; x++) {
    for (int y = 1; y <= binsy; y++) {
      const double bin_data = data.GetBinContent(x, y);
      const double bin_pred = pred.GetBinContent(x, y);
      if (bin_data != 0) [[likely]]
        chi2 +=
            (bin_pred - bin_data) + bin_data * std::log(bin_data / bin_pred);
      else
        chi2 += bin_pred;
    }
  }
  return 2 * chi2;
}
