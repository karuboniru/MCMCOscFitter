#pragma once
// c++ header
#include <memory>

// root headers
#include <TH3D.h>

// project headers


// interpolated flux input from HKKM
// always assumes 3D input

// flux to be saved in TH3D (log10 E, cos theta, phi) -> (nue, nuebar, numu,
// numubar)


class HondaFlux {
public:
  HondaFlux(const char *fluxfile);
  ~HondaFlux();

  // set the flux file

  // get the flux at a given energy and direction
  // energy in GeV, direction in cos(theta), phi
  // returns the flux in units of GeV^-1 cm^-2 s^-1 sr^-1
  double GetFlux(double energy, double costheta, double phi, int pdg);

private:
  void LoadFluxFile(const char *fluxfile);
  // flux histograms
  std::unique_ptr<TH3D> fFluxHist_nue;
  std::unique_ptr<TH3D> fFluxHist_nuebar;
  std::unique_ptr<TH3D> fFluxHist_numu;
  std::unique_ptr<TH3D> fFluxHist_numubar;

  // flux file loaded
  bool fFluxFileLoaded{false};
};