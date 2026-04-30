#include "SimpleDataHist.h"
#include <TFile.h>
#include <cmath>
#include <numeric>

void SimpleDataHist::SaveAs(const char *filename) const {
  TFile file(filename, "recreate");
  hist_numu().Write("numu");
  hist_nue().Write("nue");
  hist_numubar().Write("numubar");
  hist_nuebar().Write("nuebar");
  if (data_nc.has_value()) {
    data_nc->to_th2d(Ebins, costheta_bins).Write("nc");
  }
  file.Write();
  file.Close();
}

void SimpleDataHist::LoadFrom(const char *filename) {
  TFile file(filename, "read");
  auto *h_numu    = dynamic_cast<TH2D *>(file.Get("numu"));
  auto *h_nue     = dynamic_cast<TH2D *>(file.Get("nue"));
  auto *h_numubar = dynamic_cast<TH2D *>(file.Get("numubar"));
  auto *h_nuebar  = dynamic_cast<TH2D *>(file.Get("nuebar"));
  if (!h_numu || !h_nue || !h_numubar || !h_nuebar) {
    file.Close();
    return;
  }
  Ebins.resize(h_numu->GetNbinsX() + 1);
  for (int i = 1; i <= h_numu->GetNbinsX() + 1; ++i)
    Ebins[static_cast<size_t>(i - 1)] = h_numu->GetXaxis()->GetBinLowEdge(i);
  costheta_bins.resize(h_numu->GetNbinsY() + 1);
  for (int i = 1; i <= h_numu->GetNbinsY() + 1; ++i)
    costheta_bins[static_cast<size_t>(i - 1)] = h_numu->GetYaxis()->GetBinLowEdge(i);

  data_numu    = PodHist2D<double>::from_th2d(*h_numu);
  data_nue     = PodHist2D<double>::from_th2d(*h_nue);
  data_numubar = PodHist2D<double>::from_th2d(*h_numubar);
  data_nuebar  = PodHist2D<double>::from_th2d(*h_nuebar);
  file.Close();
}

void SimpleDataHist::Round() {
  for (size_t i = 0; i < data_numu.size(); ++i) {
    data_numu.data[i]    = std::round(data_numu.data[i]);
    data_nue.data[i]     = std::round(data_nue.data[i]);
    data_numubar.data[i] = std::round(data_numubar.data[i]);
    data_nuebar.data[i]  = std::round(data_nuebar.data[i]);
  }
}

double SimpleDataHist::total_numu()    const { return std::accumulate(data_numu.data.begin(),    data_numu.data.end(),    0.0); }
double SimpleDataHist::total_numubar() const { return std::accumulate(data_numubar.data.begin(), data_numubar.data.end(), 0.0); }
double SimpleDataHist::total_nue()     const { return std::accumulate(data_nue.data.begin(),     data_nue.data.end(),     0.0); }
double SimpleDataHist::total_nuebar()  const { return std::accumulate(data_nuebar.data.begin(),  data_nuebar.data.end(),  0.0); }
