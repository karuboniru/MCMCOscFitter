#include "genie_xsec.h"

#include <TF1.h>
#include <exception>
#include <format>
#include <iostream>
#include <print>
#include <stdexcept>

#include <TSpline.h>

genie_xsec::genie_xsec(const char *splinefile) { LoadSplineFile(splinefile); }

genie_xsec::~genie_xsec() {}

void genie_xsec::LoadSplineFile(const char *splinefile) {
  spline_file = std::make_unique<TFile>(splinefile);
}

double genie_xsec::GetXsec(double energy, int nud, int tar, bool is_cc) {

  if (!spline_file) {
    std::cerr << "Spline file not loaded" << std::endl;
    throw std::runtime_error("");
  }

  {
    auto &&iter = fXsecHist.find(std::make_tuple(nud, tar, is_cc));
    if (iter != fXsecHist.end()) {
      // return fXsecHist[std::make_tuple(nud, tar)]->Eval(energy);
      return iter->second.Eval(energy);
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

  // std::string hist_name = std::string(nu_name) + "_" + target_name +
  // "/tot_cc";
  auto hist_name = std::format("{}_{}/{}", nu_name, target_name,
                               is_cc ? "tot_cc" : "tot_nc");
  auto &&spline_graph =
      dynamic_cast<TGraph *>(spline_file->Get(hist_name.c_str()));
  if (!spline_graph) {
    std::cerr << "Failed to load spline graph: \t" << hist_name << std::endl;
    throw std::runtime_error("");
  }
  auto &&[iter, success] = fXsecHist.try_emplace(
      std::make_tuple(nud, tar, is_cc), TSpline3{"", spline_graph});
  return iter->second.Eval(energy);
}

TH1D genie_xsec::GetXsecHistMixture(
    std::vector<double> energy_bins, int nud,
    const std::vector<std::pair<int, double>> &mix_target, bool is_cc) {
  TH1D ret{"", "", static_cast<int>(energy_bins.size()) - 1,
           energy_bins.data()};
  TF1 f{"",
        [&](double *x, double *) -> double {
          double ret{};
          for (auto &&[tar, mix] : mix_target) {
            auto var = GetXsec(x[0], nud, tar, is_cc) * mix;
            var = std::max(var, 0.0);
            ret += var;
          }
          return ret;
        },
        energy_bins.front(), energy_bins.back(), 0};
  for (int i = 1; i <= ret.GetNbinsX(); ++i) {
    ret.SetBinContent(
        i, f.Integral(ret.GetBinLowEdge(i), ret.GetBinLowEdge(i + 1)) /
               (ret.GetBinWidth(i)));
  }
  return ret;
}

// TH1D genie_xsec::GetXsecHistMixture

genie_xsec xsec_input([]() -> const char * {
  auto var = std::getenv("XSEC_FILE");
  if (var)
    return var;

  return DATA_PATH "/data/total_xsec.root";
}()

);
