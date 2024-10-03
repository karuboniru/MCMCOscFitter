#include "Prob3ppOscillation.h"
#include "BargerPropagator.h"

[[gnu::const]]
std::array<std::array<double, 3>, 3>
Prob3ppOscillation::GetProb(int flavor, double E, double costheta) const {
  flavor = flavor / abs(flavor);
  BargerPropagator b;
  b.SetMNS(GetT12(), GetT13(), GetT23(), GetDm2(), GetDM2(),
           GetDeltaCP() /*delta cp*/, E, true, flavor);
  b.DefinePath(costheta, 25);
  b.propagate(flavor);
  std::array<std::array<double, 3>, 3> ret{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ret[i][j] = b.GetProb(flavor * (i + 1), flavor * (j + 1));
    }
  }
  return ret;
}

[[gnu::const]]
std::array<std::array<TH2D, 2>, 2> Prob3ppOscillation::GetProb_Hist(
    std::vector<double> Ebin, std::vector<double> costhbin, int flavor) const {
  std::array<std::array<TH2D, 2>, 2> ret{};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      ret[i][j] = TH2D("", "", Ebin.size() - 1, Ebin.data(),
                       costhbin.size() - 1, costhbin.data());
    }
  }

  for (int i = 1; i <= ret[0][0].GetNbinsX(); i++) {
    const double emin = ret[0][0].GetXaxis()->GetBinLowEdge(i);
    const double emax = ret[0][0].GetXaxis()->GetBinUpEdge(i);
    for (int j = 1; j <= ret[0][0].GetNbinsY(); j++) {
      const double costh = ret[0][0].GetYaxis()->GetBinCenter(j);
      auto prob = GetProb(flavor, (emin + emax) / 2, costh);
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 2; ++l) {
          ret[k][l].SetBinContent(i, j, prob[k][l]);
          // std::cout << prob[k][l] << std::endl;
        }
      }
    }
  }

  return ret;
}