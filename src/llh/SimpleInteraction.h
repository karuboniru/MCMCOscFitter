#pragma once

#include "OscillationParameters.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataPoint.h"
#include "genie_xsec.h"
#include "hondaflux.h"
#include <memory>

#include "DataSet.h"

// extern HondaFlux flux_input;
// extern genie_xsec xsec_input;

class SimpleInteraction : public OscillationParameters {
public:
  SimpleInteraction() = default;
  void proposeStep();
  double GetLogLikelihoodAgainstData(const DataSet<SimpleDataPoint> &dataset) const;

private:
  std::shared_ptr<Prob3ppOscillation> propagator;
  double weight_int{};
};

