#include "hondaflux2d.h"
#include <TF1.h>
#include <TH2.h>
#include <TH3.h>
#include <TSpline.h>
#include <cassert>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <TMath.h>
#include <string>

namespace {
// const unsigned int kGHnd3DNumCosThetaBins = 20;
const double kGHnd3DCosThetaMin = -1.0;
// const double kGHnd3DCosThetaMax = 1.0;
// const unsigned int kGHnd3DNumLogEvBins = 101;
// const unsigned int kGHnd3DNumLogEvBinsPerDecade = 20;
} // namespace

size_t HondaFlux2D::to_costh_bin(double costh) {
  return (costh - kGHnd3DCosThetaMin) * 10;
}

HondaFlux2D::HondaFlux2D(const char *fluxfile) {
  std::fstream flux_file_in{fluxfile, std::ios::in};
  if (!flux_file_in.is_open()) {
    std::cerr << "Error: cannot open flux file " << fluxfile << "\n";
    return;
  }
  size_t current_cth_bin{20};
  double e_min{+INFINITY}, e_max{};
  for (std::string line{}; std::getline(flux_file_in, line);) {
    if (line[0] == 'a') {
      if (current_cth_bin == 0) {
        throw std::runtime_error{"Error: unexpected end of file"};
      }
      current_cth_bin--;
      std::getline(flux_file_in, line);
      continue;
    }
    std::istringstream iss{line};
    double energy{};
    double flux_nue{}, flux_nuebar{}, flux_numu{}, flux_numubar{};
    iss >> energy >> flux_numu >> flux_numubar >> flux_nue >> flux_nuebar;
    if (iss.fail()) {
      throw std::runtime_error{"Error: failed to parse line"};
    }
    graph_flux_numu[current_cth_bin].AddPoint(energy, flux_numu);
    graph_flux_numubar[current_cth_bin].AddPoint(energy, flux_numubar);
    graph_flux_nue[current_cth_bin].AddPoint(energy, flux_nue);
    graph_flux_nuebar[current_cth_bin].AddPoint(energy, flux_nuebar);
    e_min = std::min(e_min, energy);
    e_max = std::max(e_max, energy);
  }
  for (size_t i = 0; i < 20; i++) {
    // flux_numu[i] = TSpline3{"flux_numu", &graph_flux_numu[i]};
    // flux_numubar[i] = TSpline3{"flux_numubar", &graph_flux_numubar[i]};
    // flux_nue[i] = TSpline3{"flux_nue", &graph_flux_nue[i]};
    // flux_nuebar[i] = TSpline3{"flux_nuebar", &graph_flux_nuebar[i]};
    flux_numu[i] = TF1{
        "flux_numu",
        [spl_numu = TSpline3{"spl_numu", &graph_flux_numu[i]}](
            const double *x, const double *) { return spl_numu.Eval(x[0]); },
        e_min, e_max, 0};
    flux_numubar[i] = TF1{
        "flux_numubar",
        [spl_numubar = TSpline3{"spl_numubar", &graph_flux_numubar[i]}](
            const double *x, const double *) { return spl_numubar.Eval(x[0]); },
        e_min, e_max, 0};
    flux_nue[i] =
        TF1{"flux_nue",
            [spl_nue = TSpline3{"spl_nue", &graph_flux_nue[i]}](
                const double *x, const double *) { return spl_nue.Eval(x[0]); },
            e_min, e_max, 0};
    flux_nuebar[i] = TF1{
        "flux_nuebar",
        [spl_nuebar = TSpline3{"spl_nuebar", &graph_flux_nuebar[i]}](
            const double *x, const double *) { return spl_nuebar.Eval(x[0]); },
        e_min, e_max, 0};
  }
  fFluxFileLoaded = true;
}

TH2D HondaFlux2D::GetFlux_Hist(std::vector<double> Ebins,
                               std::vector<double> costh_bins, int pdg) {
  if (!fFluxFileLoaded) {
    throw std::runtime_error{"Error: flux file not loaded"};
  }
  TH2D flux_hist{"flux_hist",
                 "flux_hist",
                 static_cast<int>(Ebins.size() - 1),
                 Ebins.data(),
                 static_cast<int>(costh_bins.size() - 1),
                 costh_bins.data()};

  for (size_t i = 0; i < flux_hist.GetNbinsX(); i++) {
    for (size_t j = 0; j < flux_hist.GetNbinsY(); j++) {
      double e_low = flux_hist.GetXaxis()->GetBinLowEdge(i + 1);
      double e_up = flux_hist.GetXaxis()->GetBinUpEdge(i + 1);
      assert(e_low < e_up);
      assert(j < 20);
      // assert(to_costh_bin())
      double cth_bin_center = flux_hist.GetYaxis()->GetBinCenter(j + 1);
      assert(to_costh_bin(cth_bin_center) == j);
      auto &flux_object = [&]() -> TF1 & {
        switch (pdg) {
        case 12:
          return flux_nue[j];
        case -12:
          return flux_nuebar[j];
        case 14:
          return flux_numu[j];
        case -14:
          return flux_numubar[j];
        default:
          throw std::runtime_error{"Error: unknown pdg code"};
        }
      }();
      flux_hist.SetBinContent(i + 1, j + 1,
                              flux_object.Integral(e_low, e_up) * (2 * M_PI));
    }
  }
  return flux_hist;
}

HondaFlux2D::~HondaFlux2D() = default;

inline HondaFlux2D flux_input(std::getenv("FLUX_FILE_2D"));

