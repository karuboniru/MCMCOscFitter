#pragma once

#include "SimpleDataPoint.h"
#include <functional>

namespace {
double Emin = 3, Emax = 10;
}

class NeutrinoState : public SimpleDataPoint {
public:
  ~NeutrinoState() = default;

  NeutrinoState(double E_, double costheta_, double phi_, int flavor_)
      : SimpleDataPoint(E_, costheta_, phi_, flavor_) {};
  NeutrinoState(const NeutrinoState &) = default;
  NeutrinoState(NeutrinoState &&) = default;
  NeutrinoState &operator=(const NeutrinoState &) = default;
  NeutrinoState &operator=(NeutrinoState &&) = default;

  // default constructor, should not be used
  NeutrinoState() = default;

  void proposeStep();
  double GetLogLikelihood() const;
};
