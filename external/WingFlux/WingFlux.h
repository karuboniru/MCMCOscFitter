#pragma once
#include <TH2.h>

class WingFlux {
public:
  WingFlux(const char *fluxfile);
  WingFlux() = delete;
  WingFlux(const WingFlux &) = delete;
  WingFlux &operator=(const WingFlux &) = delete;
  ~WingFlux();


  TH2D GetFlux_Hist(int pdg) const;

private:
  // 
  TH2D numu;
  TH2D numubar;
  TH2D nue;
  TH2D nuebar;
};

extern const WingFlux wingflux;