#include "NeutrinoState.h"
#include <TMath.h>
#include <TRandom.h>
#include <cmath>

void NeutrinoState::proposeStep() {
  E = gRandom->Uniform(Emin, Emax);
  costheta = gRandom->Uniform(-1, 1);
  phi = gRandom->Uniform(0, 2 * TMath::Pi());
  switch (gRandom->Integer(4)) {
  case 0:
    flavor = -14;
    break;
  case 1:
    flavor = 14;
    break;
  case 2:
    flavor = 12;
    break;
  case 3:
    flavor = -12;
    break;
  }
}

double NeutrinoState::GetLogLikelihood() const { return 0; }
