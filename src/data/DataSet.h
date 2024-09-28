#pragma once

#include "StateI.h"
#include <algorithm>
#include <numeric>
#include <vector>

template <std::derived_from<StateI> T>
class DataSet : virtual public StateI, public std::vector<T> {
public:
  using std::vector<T>::vector;
  virtual void proposeStep() override {
    // std::ranges::for_each(*this, [](T &point) { point.proposeStep(); });
    std::for_each(this->begin(), this->end(),
                  [](T &point) { point.proposeStep(); });
  }
  virtual double GetLogLikelihood() const override {
    return std::accumulate(this->cbegin(), this->cend(), 0.0,
                           [](double sum, const T &point) {
                             return sum + point.GetLogLikelihood();
                           });
  }
};
