#pragma once

#include "mcmc_concepts.h"
#include <algorithm>
#include <numeric>
#include <vector>

template <mcmc_concepts::MCMCState T>
class DataSet : public std::vector<T> {
public:
  using std::vector<T>::vector;
  void proposeStep() {
    std::for_each(this->begin(), this->end(),
                  [](T &point) { point.proposeStep(); });
  }
  double GetLogLikelihood() const {
    return std::accumulate(this->cbegin(), this->cend(), 0.0,
                           [](double sum, const T &point) {
                             return sum + point.GetLogLikelihood();
                           });
  }
};
