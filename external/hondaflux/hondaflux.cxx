#include <cassert>
#include <hondaflux.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <TMath.h>
#include <string>

namespace {
const unsigned int kGHnd3DNumCosThetaBins = 20;
const double kGHnd3DCosThetaMin = -1.0;
const double kGHnd3DCosThetaMax = 1.0;
const unsigned int kGHnd3DNumPhiBins = 12;
const double kGHnd3DPhiMin = 0.0;
const double kGHnd3DPhiMax = 360.0;
const unsigned int kGHnd3DNumLogEvBins = 101;
const unsigned int kGHnd3DNumLogEvBinsPerDecade = 20;
} // namespace

HondaFlux::HondaFlux(const char *fluxfile) {
  // auto fCosThetaBins    = new double [kGHnd3DNumCosThetaBins + 1];
  auto fCosThetaBins = std::make_unique<double[]>(kGHnd3DNumCosThetaBins + 1);
  // auto fNumCosThetaBins = kGHnd3DNumCosThetaBins;

  double dcostheta = (kGHnd3DCosThetaMax - kGHnd3DCosThetaMin) /
                     (double)kGHnd3DNumCosThetaBins;

  for (unsigned int i = 0; i <= kGHnd3DNumCosThetaBins; i++) {
    fCosThetaBins[i] = kGHnd3DCosThetaMin + i * dcostheta;
  }

  //
  // phi
  //

  // auto fPhiBins    = new double [kGHnd3DNumPhiBins + 1];
  auto fPhiBins = std::make_unique<double[]>(kGHnd3DNumPhiBins + 1);
  // auto fNumPhiBins = kGHnd3DNumPhiBins;

  double d2r = M_PI / 180.;

  double dphi =
      d2r * (kGHnd3DPhiMax - kGHnd3DPhiMin) / (double)kGHnd3DNumPhiBins;

  for (unsigned int i = 0; i <= kGHnd3DNumPhiBins; i++) {
    fPhiBins[i] = kGHnd3DPhiMin + i * dphi;
  }

  //
  // log(E)
  //

  // For each costheta,phi pair there are N logarithmically spaced
  // neutrino energy values (starting at 0.1 GeV with 20 values per decade
  // up to 10000 GeV) each with corresponding flux values.
  // To construct a flux histogram, use N+1 bins from 0 up to maximum
  // value. Assume that the flux value given for E=E0 is the flux value
  // at the bin that has E0 as its upper edge.
  //
  // auto fEnergyBins = new double[kGHnd3DNumLogEvBins + 1]; // bin edges
  auto fEnergyBins = std::make_unique<double[]>(kGHnd3DNumLogEvBins + 1);
  // kGHnd3DNumLogEvBins = kGHnd3DNumLogEvBins;

  double logEmax = TMath::Log10(1.);
  double logEmin = TMath::Log10(0.1);
  double dlogE = (logEmax - logEmin) / (double)kGHnd3DNumLogEvBinsPerDecade;

  fEnergyBins[0] = 0;
  for (unsigned int i = 0; i < kGHnd3DNumLogEvBins; i++) {
    fEnergyBins[i + 1] = TMath::Power(10., logEmin + i * dlogE);
  }

  auto make_hist = [&](const char *name, const char *title) {
    return std::make_unique<TH3D>(name, title, kGHnd3DNumLogEvBins,
                                  fEnergyBins.get(), kGHnd3DNumCosThetaBins,
                                  fCosThetaBins.get(), kGHnd3DNumPhiBins,
                                  fPhiBins.get());
  };

  fFluxHist_nue = make_hist("fFluxHist_nue", "fFluxHist_nue");
  fFluxHist_nuebar = make_hist("fFluxHist_nuebar", "fFluxHist_nuebar");
  fFluxHist_numu = make_hist("fFluxHist_numu", "fFluxHist_numu");
  fFluxHist_numubar = make_hist("fFluxHist_numubar", "fFluxHist_numubar");

  LoadFluxFile(fluxfile);
}

HondaFlux::~HondaFlux() {}

void HondaFlux::LoadFluxFile(const char *fluxfile) {
  std::ifstream flux_stream(fluxfile, std::ios::in);
  size_t ienergy = 1;
  size_t iphi = 0;
  size_t icostheta =
      kGHnd3DNumCosThetaBins; // honda starts from [0.9 - 1.0], i.e. last bin.
  for (std::string line{}; std::getline(flux_stream, line);) {
    // std::cerr << line << std::endl;
    if (line[0] == 'a') {
      // a start of new block, update bin ids
      iphi++;
      if (iphi == kGHnd3DNumPhiBins + 1) {
        icostheta--;
        iphi = 1;
      }
      ienergy = 1;
      continue;
    }
    if (line[1] == 'E') {
      continue;
    }
    std::istringstream iss(line);
    double energy, fnumu, fnumubar, fnue, fnuebar;
    iss >> energy >> fnumu >> fnumubar >> fnue >> fnuebar;

    // std::cerr << ienergy << " "
    //           << fFluxHist_nue->GetXaxis()->FindBin(energy) << " "
    //           << TMath::Log10(energy) << std::endl;
    fFluxHist_nue->SetBinContent(ienergy, icostheta, iphi, fnue);
    fFluxHist_nuebar->SetBinContent(ienergy, icostheta, iphi, fnuebar);
    fFluxHist_numu->SetBinContent(ienergy, icostheta, iphi, fnumu);
    fFluxHist_numubar->SetBinContent(ienergy, icostheta, iphi, fnumubar);
    ienergy++;
  }

  fFluxFileLoaded = true;
}

double HondaFlux::GetFlux(double energy, double costheta, double phi, int pdg) {
  if (!fFluxFileLoaded) {
    std::cerr << "Flux file not loaded" << std::endl;
    throw std::invalid_argument("Flux file not loaded");
  }

  // convert energy to log10(E)
  // double logE = TMath::Log10(energy);

  // find the bin
  int ienergy = fFluxHist_nue->GetXaxis()->FindBin(energy);
  // if (ienergy > )
  int icostheta = fFluxHist_nue->GetYaxis()->FindBin(costheta);
  int iphi = fFluxHist_nue->GetZaxis()->FindBin(phi);

  // get the flux
  double flux = 0;
  switch (pdg) {
  case 12:
    flux = fFluxHist_nue->GetBinContent(ienergy, icostheta, iphi);
    break;
  case -12:
    flux = fFluxHist_nuebar->GetBinContent(ienergy, icostheta, iphi);
    break;
  case 14:
    flux = fFluxHist_numu->GetBinContent(ienergy, icostheta, iphi);
    break;
  case -14:
    flux = fFluxHist_numubar->GetBinContent(ienergy, icostheta, iphi);
    break;
  default:
    std::cerr << "Unknown PDG code: " << pdg << std::endl;
    throw std::invalid_argument("Unknown PDG code");
    break;
  }

  return flux;
}