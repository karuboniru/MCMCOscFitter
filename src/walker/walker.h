#pragma once
#include "mcmc_concepts.h"
#include "StateI.h"
#include <TRandom3.h>
#include <random>

// Template versions — used by all C++ code.
// Constrained on MCMCState concept instead of depending on StateI inheritance.

template <mcmc_concepts::MCMCState State>
bool MCMCAcceptState(const State &current, const State &next) {
  double current_logweight = current.GetLogLikelihood();
  double next_logweight = next.GetLogLikelihood();
  double log_ratio = next_logweight - current_logweight;
  if (log_ratio > 0) return true;
  auto rand = gRandom->Rndm();
  return rand < std::exp(log_ratio);
}

template <mcmc_concepts::MCMCState State>
bool MCMCAcceptState(const State &current, const State &next,
                     std::mt19937 &rng) {
  double current_logweight = current.GetLogLikelihood();
  double next_logweight = next.GetLogLikelihood();
  double log_ratio = next_logweight - current_logweight;
  if (log_ratio > 0) return true;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng) < std::exp(log_ratio);
}

/// Temperature-aware overload — divides log-ratio by temperature.
template <mcmc_concepts::MCMCState State>
bool MCMCAcceptState(const State &current, const State &next,
                     double temperature) {
  double current_logweight = current.GetLogLikelihood();
  double next_logweight = next.GetLogLikelihood();
  double log_ratio = next_logweight - current_logweight;
  if (log_ratio > 0) return true;
  return gRandom->Rndm() < std::exp(log_ratio / temperature);
}

// Non-template overloads — kept for backward compatibility with code that
// passes StateI& (primarily pybind11 runtime polymorphism).
bool MCMCAcceptState(const StateI &current, const StateI &next);
bool MCMCAcceptState(const StateI &current, const StateI &next,
                     std::mt19937 &rng);
bool MCMCAcceptState(const StateI &current, const StateI &next,
                     double temperature);
