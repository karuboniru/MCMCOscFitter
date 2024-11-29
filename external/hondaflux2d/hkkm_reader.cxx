#include "hkkm_reader.hxx"

#include <TH2.h>
#include <TMath.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
constexpr double cos_theta_min = -1;
constexpr double cos_theta_max = 1;
constexpr size_t n_costh_bins = 20;
[[maybe_unused]] constexpr double d_costh =
    (cos_theta_max - cos_theta_min) / n_costh_bins;

constexpr double logE_min = -1;
constexpr double logE_max = 4;
constexpr size_t n_logE_points = 101;
constexpr double d_logE = (logE_max - logE_min) / (n_logE_points - 1);

TH2D create_hist_model() {
  return {"", "",
          // points from HKKM is in the middle of the bin
          n_logE_points, logE_min - d_logE / 2., logE_max + d_logE / 2.,
          // same binning as HKKM for costh
          n_costh_bins, cos_theta_min, cos_theta_max};
}

std::array<TH2D, 4> read_honda_flux(const std::string &path) {
  std::array<TH2D, 4> hists{create_hist_model(), create_hist_model(),
                            create_hist_model(), create_hist_model()};
  auto &[numu, numubar, nue, nuebar] = hists;
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open file");
  }

  double costh{};
  for (std::string line{}; std::getline(file, line);) {
    if (line[0] == 'a') {
      // we are at sth like
      // average flux in [cosZ = 0.90 --  1.00, phi_Az =   0 --  30]
      // read cos and phi from it
      double cosz_low{}, cosz_high{}, phi_low{}, phi_high{};
      std::sscanf(line.c_str(),
                  "average flux in [cosZ = %lf -- %lf, phi_Az = %lf -- %lf]",
                  &cosz_low, &cosz_high, &phi_low, &phi_high);
      costh = (cosz_low + cosz_high) / 2;
      continue;
    }
    if (line[1] == 'E') {
      continue;
    }
    double E{}, flux_numu{}, flux_numubar{}, flux_nue{}, flux_nuebar{};
    std::stringstream ss(line);
    ss >> E >> flux_numu >> flux_numubar >> flux_nue >> flux_nuebar;
    auto logE = std::log10(E);
    auto ebin = numu.GetXaxis()->FindBin(logE);
    auto costhbin = numu.GetYaxis()->FindBin(costh);
    numu.SetBinContent(ebin, costhbin, flux_numu);
    numubar.SetBinContent(ebin, costhbin, flux_numubar);
    nue.SetBinContent(ebin, costhbin, flux_nue);
    nuebar.SetBinContent(ebin, costhbin, flux_nuebar);
  }
  return hists;
}

} // namespace

HKKM_READER_2D::HKKM_READER_2D(const std::string &path)
    : hists(read_honda_flux(path)) {}

size_t HKKM_READER_2D::pdg_index(int pdg) {
  switch (pdg) {
  case 14:
    return 0;
  case -14:
    return 1;
  case 12:
    return 2;
  case -12:
    return 3;
  default:
    throw std::runtime_error("Invalid pdg code");
  }
}