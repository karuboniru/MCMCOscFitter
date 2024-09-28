#pragma once

#include "SimpleDataPoint.h"
#include "StateI.h"
#include <functional>

namespace {
double Emin = 3, Emax = 10;
}

class NeutrinoState : virtual public SimpleDataPoint {
public:
  ~NeutrinoState() = default;

  // NeutrinoState(std::function<double(double, double, double, int)> m_w)
  //     : weight_calculator(m_w) {};
  NeutrinoState(double E_, double costheta_, double phi_, int flavor_)
      : SimpleDataPoint(E_, costheta_, phi_, flavor_) {};
  NeutrinoState(const NeutrinoState &) = default;
  NeutrinoState(NeutrinoState &&) = default;
  NeutrinoState &operator=(const NeutrinoState &) = default;
  NeutrinoState &operator=(NeutrinoState &&) = default;

  // default constructor, should not be used
  NeutrinoState() = default;

  // double E, costheta, phi;
  // int flavor;
  // double weight;

  // std::function<double(double, double, double, int)> weight_calculator;

  virtual void proposeStep() override;
  virtual double GetLogLikelihood() const override;
};
