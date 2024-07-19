#pragma once

#include "ModelDataLLH.h"
#include "OscillationParameters.h"
#include "SimpleDataPoint.h"
#include "genie_xsec.h"
#include "hondaflux.h"

class SimpleInteraction : public Prob3ppOscillation, public ModelDataLLH {
public:
  SimpleInteraction(): Prob3ppOscillation(), ModelDataLLH() {}
  virtual void proposeStep() override;
  // virtual double GetLogLikelihood() const override;
  virtual double
  GetLogLikelihoodAgainstData(const StateI &dataset) const override;

private: 
  double weight_int{};

private:
  static HondaFlux flux_input;
  static genie_xsec xsec_input;
};
