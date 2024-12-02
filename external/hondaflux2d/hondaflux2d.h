#pragma once
// c++ header
#include <TF1.h>
#include <TGraph.h>
#include <TH2.h>
#include <interpolation.hxx>
#include <memory>

class TH3D;
class TH2D;

using interpolater_type = interpolate<3, 4>;

class HondaFlux2D {
public:
  HondaFlux2D(const char *fluxfile);
  ~HondaFlux2D();

  TH2D GetFlux_Hist(std::vector<double> Ebins, std::vector<double> costh_bins,
                    int pdg);

private:
  // interpolate<3, 4> numu, numubar, nue, nuebar;
  std::array<interpolater_type, 4> interp;
};

// inline HondaFlux flux_input("/var/home/yan/neutrino/honda-3d.txt");
extern HondaFlux2D flux_input;
