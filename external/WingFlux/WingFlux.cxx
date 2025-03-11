#include "WingFlux.h"
#include <binning_tool.hpp>
#include <fstream>
#include <generator>
#include <iostream>

std::generator<std::string> read_lines(std::string filename) {
  std::ifstream file(filename);
  std::string line;
  while (std::getline(file, line)) {
    co_yield line;
  }
}

TH2D hist_model() {
  TH1::AddDirectory(false);
  static auto costh_bins = linspace(-1., 1., 401);
  static auto ebins = logspace(0.1, 20., 401);
  return {"",
          "",
          (int)(ebins.size() - 1),
          ebins.data(),
          (int)(costh_bins.size() - 1),
          costh_bins.data()};
}

void re_normalize_hist(TH2D &hist) {
  for (int ix = 1; ix <= hist.GetNbinsX(); ++ix) {
    double dx = hist.GetXaxis()->GetBinWidth(ix);
    for (int iy = 1; iy <= hist.GetNbinsY(); ++iy) {
      double dy = hist.GetYaxis()->GetBinWidth(iy);
      double content = hist.GetBinContent(ix, iy);
      hist.SetBinContent(ix, iy, content * dx * dy * 2 * M_PI);
    }
  }
}

WingFlux::WingFlux(const char *fluxfile)
    : numu(hist_model()), numubar(hist_model()), nue(hist_model()),
      nuebar(hist_model()) {
  for (auto line : read_lines(fluxfile) | std::views::drop(1)) {
    // tokenize the line with \t or space as delimiter
    auto tokens = line | std::views::split('\t') |
                  std::views::transform([](auto &&s) {
                    auto str = std::string(s.begin(), s.end());
                    return std::stod(str);
                  }) |
                  std::ranges::to<std::vector>();
    if (tokens.size() != 6) {
      std::cerr << "Invalid line: " << line << std::endl;
      continue;
    }
    auto E_val = tokens[0];
    auto E = numu.GetXaxis()->FindBin(E_val);
    auto zen_val = tokens[1];
    auto zen = numu.GetYaxis()->FindBin(zen_val);
    auto numu_flux = tokens[4];
    auto numubar_flux = tokens[5];
    auto nue_flux = tokens[2];
    auto nuebar_flux = tokens[3];
    numu.SetBinContent(E, zen, numu_flux);
    numubar.SetBinContent(E, zen, numubar_flux);
    nue.SetBinContent(E, zen, nue_flux);
    nuebar.SetBinContent(E, zen, nuebar_flux);
  }
  re_normalize_hist(numu);
  re_normalize_hist(numubar);
  re_normalize_hist(nue);
  re_normalize_hist(nuebar);
}

TH2D WingFlux::GetFlux_Hist(int pdg) const {
  switch (pdg) {
  case 12:
    return nue;
  case -12:
    return nuebar;
  case 14:
    return numu;
  case -14:
    return numubar;
  default:
    throw std::invalid_argument("Invalid PDG code");
  }
}

WingFlux::~WingFlux() = default;

const WingFlux wingflux([]() -> const char * {
  auto env_str = std::getenv("FLUX_FILE_2D");
  if (env_str)
    return env_str;
  return DATA_PATH "/data/int_table_400x400_1D_IP_logE.dat";
}());