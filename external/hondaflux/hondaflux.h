#pragma once
// c++ header
#include <TH2.h>
#include <memory>

class TH3D;
class TH2D;
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

  double GetFlux(double energy, double costheta, int pdg);

  TH2D GetFlux_Hist(std::vector<double> Ebins, std::vector<double> costh_bins,
                    int pdg);

private:
  void LoadFluxFile(const char *fluxfile);
  // flux histograms
  std::unique_ptr<TH3D> fFluxHist_nue;
  std::unique_ptr<TH3D> fFluxHist_nuebar;
  std::unique_ptr<TH3D> fFluxHist_numu;
  std::unique_ptr<TH3D> fFluxHist_numubar;

  std::unique_ptr<TH2D> fFluxHist_2d_nue;
  std::unique_ptr<TH2D> fFluxHist_2d_nuebar;
  std::unique_ptr<TH2D> fFluxHist_2d_numu;
  std::unique_ptr<TH2D> fFluxHist_2d_numubar;

  // flux file loaded
  bool fFluxFileLoaded{false};
};

// inline HondaFlux flux_input("/var/home/yan/neutrino/honda-3d.txt");
inline HondaFlux flux_input(std::getenv("FLUX_FILE"));
