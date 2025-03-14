#pragma once

#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TSpline.h>
#include <cstdlib>
#include <map>
#include <memory>

class genie_xsec {
public:
  genie_xsec(const char *splinefile);
  ~genie_xsec();

  double GetXsec(double energy, int nud, int tar, bool is_cc = true);
  // TSpline3 GetXsecSpl(int nud, int tar, bool is_cc);

  // TH1D GetXsecHist(std::vector<double> energy_bins, int nud, int tar);
  TH1D GetXsecHistMixture(std::vector<double> energy_bins, int nud,
                          const std::vector<std::pair<int, double>> &mix_target,
                          bool is_cc = true);

private:
  void LoadSplineFile(const char *splinefile);
  std::unique_ptr<TFile> spline_file;
  std::map<std::tuple<int, int, bool>, TSpline3> fXsecHist;
};

// inline genie_xsec xsec_input(
//     "/var/home/yan/neutrino/spline/full/genie_spline_3_0_2/G18_02b_02_11b/"
//     "3.04.02-routine_validation_01-xsec_vA/total_xsec.root");
extern genie_xsec xsec_input;