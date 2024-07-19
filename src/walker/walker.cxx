#include "walker.h"

#include <TRandom3.h>
#include <iostream>

bool MCMCAcceptState(const StateI &current, const StateI &next) {
  double current_logweight = current.GetLogLikelihood();
  double next_logweight = next.GetLogLikelihood();

  // double ratio = next_weight / current_weight;
  double log_ratio = next_logweight - current_logweight;
  // std::cerr << "log_ratio: " << log_ratio << std::endl;
  if (log_ratio > 0)
    return true;
  auto rand = gRandom->Rndm();
  return rand < exp(log_ratio);
}