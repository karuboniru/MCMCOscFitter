#include "walker.h"

#include <TRandom3.h>
#include <iostream>
#include <cmath>

bool MCMCAcceptState(const StateI &current, const StateI &next) {
  double current_logweight = current.GetLogLikelihood();
  double next_logweight = next.GetLogLikelihood();

  double log_ratio = next_logweight - current_logweight;
  if (log_ratio > 0)
    return true;
  auto rand = gRandom->Rndm();
  return rand < std::exp(log_ratio);
}

bool MCMCAcceptState(const StateI &current, const StateI &next, std::mt19937 &rng) {
  double current_logweight = current.GetLogLikelihood();
  double next_logweight = next.GetLogLikelihood();

  double log_ratio = next_logweight - current_logweight;
  if (log_ratio > 0)
    return true;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng) < std::exp(log_ratio);
}
