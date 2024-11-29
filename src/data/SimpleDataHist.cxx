#include "SimpleDataHist.h"
#include <TFile.h>

void SimpleDataHist::SaveAs(const char *filename) const {
  TFile file(filename, "recreate");
  hist_numu.Write("numu");
  hist_nue.Write("nue");
  hist_numubar.Write("numubar");
  hist_nuebar.Write("nuebar");
  file.Write();
  file.Close();
}

void SimpleDataHist::LoadFrom(const char *filename) {
  TFile file(filename, "read");
  hist_numu = *dynamic_cast<TH2D *>(file.Get("numu"));
  hist_nue = *dynamic_cast<TH2D *>(file.Get("nue"));
  hist_numubar = *dynamic_cast<TH2D *>(file.Get("numubar"));
  hist_nuebar = *dynamic_cast<TH2D *>(file.Get("nuebar"));
}

void SimpleDataHist::Round() {
  auto round_hist_2D = [](TH2D &hist) {
#pragma omp parallel for
    for (int i = 1; i <= hist.GetNbinsX(); i++) {
      for (int j = 1; j <= hist.GetNbinsY(); j++) {
        hist.SetBinContent(i, j, std::round(hist.GetBinContent(i, j)));
      }
    }
  };
  round_hist_2D(hist_numu);
  round_hist_2D(hist_nue);
  round_hist_2D(hist_numubar);
  round_hist_2D(hist_nuebar);
}