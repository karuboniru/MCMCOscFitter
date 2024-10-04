#pragma once
// c++ header
#include <TF1.h>
#include <TGraph.h>
#include <TH2.h>
#include <memory>

class TH3D;
class TH2D;
// project headers

// interpolated flux input from HKKM
// always assumes 3D input

// flux to be saved in TH3D (log10 E, cos theta, phi) -> (nue, nuebar, numu,
// numubar)

class HondaFlux2D {
public:
  HondaFlux2D(const char *fluxfile);
  ~HondaFlux2D();

  static size_t to_costh_bin(double costh);

  TH2D GetFlux_Hist(std::vector<double> Ebins, std::vector<double> costh_bins,
                    int pdg);

private:
  std::array<TF1, 20> flux_numu;
  std::array<TF1, 20> flux_numubar;
  std::array<TF1, 20> flux_nue;
  std::array<TF1, 20> flux_nuebar;

  std::array<TGraph, 20> graph_flux_numu;
  std::array<TGraph, 20> graph_flux_numubar;
  std::array<TGraph, 20> graph_flux_nue;
  std::array<TGraph, 20> graph_flux_nuebar;
  // flux file loaded
  bool fFluxFileLoaded{false};
};

// inline HondaFlux flux_input("/var/home/yan/neutrino/honda-3d.txt");
extern HondaFlux2D flux_input;
