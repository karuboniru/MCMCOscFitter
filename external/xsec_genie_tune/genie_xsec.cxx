#include "genie_xsec.h"

#include <exception>
#include <iostream>
#include <stdexcept>

#include <TSpline.h>

genie_xsec::genie_xsec(const char *splinefile) { LoadSplineFile(splinefile); }

genie_xsec::~genie_xsec() {}

void genie_xsec::LoadSplineFile(const char *splinefile) {
  spline_file = std::make_unique<TFile>(splinefile);
}

double genie_xsec::GetXsec(double energy, int nud, int tar) {

  if (!spline_file) {
    std::cerr << "Spline file not loaded" << std::endl;
    throw std::runtime_error("");
  }

  {
    auto &&iter = fXsecHist.find(std::make_tuple(nud, tar));
    if (iter != fXsecHist.end()) {
      // return fXsecHist[std::make_tuple(nud, tar)]->Eval(energy);
      return TSpline3{"", iter->second}.Eval(energy);
    }
  }
  const char *target_name, *nu_name;

  switch (nud) {
  case 12:
    nu_name = "nu_e";
    break;
  case 14:
    nu_name = "nu_mu";
    break;
  case -12:
    nu_name = "nu_e_bar";
    break;
  case -14:
    nu_name = "nu_mu_bar";
    break;
  default:
    std::cerr << "Unexpected nu flavor: \t" << nud << std::endl;
    throw std::invalid_argument("");
    break;
  }

  switch (tar) {
  case 1000000010:
  case 2212:
    target_name = "H1";
    break;
  case 1000060120:
    target_name = "C12";
    break;
  default:
    std::cerr << "Unexpected nucleon flavor: \t" << tar << std::endl;
    throw std::invalid_argument("");
    break;
  }

  std::string hist_name = std::string(nu_name) + "_" + target_name + "/tot_cc";
  auto &&spline_graph =
      dynamic_cast<TGraph *>(spline_file->Get(hist_name.c_str()));
  if (!spline_graph) {
    std::cerr << "Failed to load spline graph: \t" << hist_name << std::endl;
    throw std::runtime_error("");
  }
  fXsecHist[std::make_tuple(nud, tar)] = spline_graph;
  return TSpline3{"", spline_graph}.Eval(energy);
}

TH1D genie_xsec::GetXsecHist(std::vector<double> energy_bins, int nud,
                             int tar) {
  TH1D ret{"", "", static_cast<int>(energy_bins.size()) - 1,
           energy_bins.data()};
  for (int i = 1; i <= ret.GetNbinsX(); ++i) {
    ret.SetBinContent(i, GetXsec(ret.GetBinCenter(i), nud, tar));
  }
  return ret;
}