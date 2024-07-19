#pragma once

#include <TFile.h>
#include <TGraph.h>
#include <map>
#include <memory>

class genie_xsec {
public:
  genie_xsec(const char *splinefile);
  ~genie_xsec();

  double GetXsec(double energy, int nud, int tar);

private:
  void LoadSplineFile(const char *splinefile);
  std::unique_ptr<TFile> spline_file;
  std::map<std::tuple<int, int>, TGraph *> fXsecHist;
};