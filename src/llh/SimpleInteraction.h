#pragma once

#include "ModelDataLLH.h"
#include "OscillationParameters.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataPoint.h"
#include "genie_xsec.h"
#include "hondaflux.h"
#include <memory>

#include "DataSet.h"

// extern HondaFlux flux_input;
// extern genie_xsec xsec_input;

class SimpleInteraction : public OscillationParameters, public ModelDataLLH<DataSet<SimpleDataPoint>> {
public:
  SimpleInteraction() : ModelDataLLH() {}
  void proposeStep() override;
  double GetLogLikelihoodAgainstData(const DataSet<SimpleDataPoint> &dataset) const override;

private:
  std::shared_ptr<Prob3ppOscillation> propagator;
  double weight_int{};
};

