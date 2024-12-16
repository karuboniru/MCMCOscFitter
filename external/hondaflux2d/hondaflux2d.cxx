#include "hondaflux2d.h"
#include "hkkm_reader.hxx"
#include <TF1.h>
#include <TF2.h>
#include <TH2.h>
#include <TH3.h>
#include <TSpline.h>
#include <cassert>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <print>
#include <sstream>
#include <stdexcept>

#include <TMath.h>
#include <string>
#include <unistd.h>

namespace {
constexpr size_t n_logE_points = 101;
constexpr size_t n_costh_bins = 20;
constexpr size_t n_costh_points = n_costh_bins + 1;
// constexpr size_t n_phi_bins = 12;
// constexpr size_t n_phi_points = n_phi_bins + 1;

constexpr axis_object logE_points{
    .min = -1, .max = 4, .n_points = n_logE_points};
constexpr axis_object costh_points{
    .min = -1, .max = 1, .n_points = n_costh_points};
// constexpr axis_object phi_points{
//     .min = 0, .max = M_PI * 2, .n_points = n_phi_points};

size_t pdg2idx(int pdg) {
  switch (pdg) {
  case 12:
    return 0;
  case 14:
    return 1;
  case -12:
    return 2;
  case -14:
    return 3;
  default:
    throw std::invalid_argument("Invalid PDG code");
  }
}
constexpr std::array<int, 4> pdg_list{12, 14, -12, -14};

HKKM_READER_2D reader([]() -> const char * {
  auto env_str = std::getenv("FLUX_FILE_2D");
  if (env_str)
    return env_str;
  return DATA_PATH "/data/honda-2d.solmin.txt";
}());
} // namespace

HondaFlux2D::HondaFlux2D(const char *fluxfile)
    : interp{interpolater_type{{logE_points, costh_points}},
             interpolater_type{{logE_points, costh_points}},
             interpolater_type{{logE_points, costh_points}},
             interpolater_type{{logE_points, costh_points}}} {
  HKKM_READER_2D reader(fluxfile);

  for (const auto pdg : pdg_list) {
    auto &flux_hist = reader[pdg];
    auto &interop_obj = interp[pdg2idx(pdg)];
    for (size_t i = 0; i < n_logE_points; ++i) {
      double cdf_along_costh = 0;
      interop_obj[{i, 0}] = 0;
      for (size_t j = 0; j < n_costh_bins; ++j) {
        cdf_along_costh += flux_hist.GetBinContent(i + 1, j + 1);
        interop_obj[{i, j + 1}] = cdf_along_costh * 0.1;
      }
    }
  }
}

TH2D HondaFlux2D::GetFlux_Hist(std::vector<double> Ebins,
                               std::vector<double> costh_bins, int pdg) {
  TH2D ret("", "", Ebins.size() - 1, Ebins.data(), costh_bins.size() - 1,
           costh_bins.data());
  auto &interp_obj = interp[pdg2idx(pdg)];
  TF2 f(
      "f",
      [&](double *x, double *) {
        double &E = x[0];
        double costh = x[1];
        double logE = log10(E);
        return interp_obj.do_interpolation({logE, costh}, {false, true}) * 2 *
               M_PI;
      },
      0.1, 1e4, -1, 1, 0);
  // #pragma omp parallel for
  for (int i = 0; i < ret.GetNbinsX(); i++) {
    auto emin = ret.GetXaxis()->GetBinLowEdge(i + 1);
    auto emax = ret.GetXaxis()->GetBinUpEdge(i + 1);
    // #pragma omp parallel for
    for (int j = 0; j < ret.GetNbinsY(); j++) {
      auto costh_min = ret.GetYaxis()->GetBinLowEdge(j + 1);
      auto costh_max = ret.GetYaxis()->GetBinUpEdge(j + 1);
      auto integration = f.Integral(emin, emax, costh_min, costh_max);
      ret.SetBinContent(i + 1, j + 1, integration);
    }
  }

  return ret;
}

HondaFlux2D::~HondaFlux2D() = default;

inline HondaFlux2D flux_input([]() -> const char * {
  auto env_str = std::getenv("FLUX_FILE_2D");
  if (env_str)
    return env_str;
  return DATA_PATH "/data/honda-2d.solmin.txt";
}());
