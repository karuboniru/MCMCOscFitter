#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mcmc {

class TemperatureLadder {
public:
  TemperatureLadder() = default;

  explicit TemperatureLadder(std::vector<double> temps)
      : values_(std::move(temps)) {
    if (values_.empty()) {
      throw std::invalid_argument(
          "TemperatureLadder: values must not be empty");
    }
    if (values_.front() != 1.0) {
      throw std::invalid_argument(
          "TemperatureLadder: first temperature must be 1.0 (cold chain)");
    }
    for (size_t i = 1; i < values_.size(); ++i) {
      if (values_[i] <= values_[i - 1]) {
        throw std::invalid_argument(
            "TemperatureLadder: temperatures must be strictly increasing");
      }
    }
  }

  // Static factories

  static TemperatureLadder geometric(size_t n, double T_min = 1.0,
                                     double T_max = 100.0) {
    std::vector<double> temps(n);
    temps[0] = T_min;
    if (n > 1) {
      double ratio = std::pow(T_max / T_min, 1.0 / (n - 1));
      for (size_t i = 1; i < n; ++i) {
        temps[i] = temps[i - 1] * ratio;
      }
    }
    return TemperatureLadder(std::move(temps));
  }

  // Access

  [[nodiscard]] size_t size() const { return values_.size(); }
  [[nodiscard]] double operator[](size_t i) const { return values_[i]; }
  [[nodiscard]] double cold() const { return values_.front(); }
  [[nodiscard]] double hottest() const { return values_.back(); }

  [[nodiscard]] double beta(size_t i) const { return 1.0 / values_[i]; }

  [[nodiscard]] std::vector<double> betas() const {
    std::vector<double> bs(size());
    for (size_t i = 0; i < size(); ++i)
      bs[i] = beta(i);
    return bs;
  }

  [[nodiscard]] const std::vector<double> &values() const { return values_; }

  // Iteration support

  [[nodiscard]] auto begin() const { return values_.begin(); }
  [[nodiscard]] auto end()   const { return values_.end(); }

private:
  std::vector<double> values_;
};

// Free-function alias (Option A entry point)
inline TemperatureLadder geometric_ladder(size_t n, double T_min = 1.0,
                                          double T_max = 100.0) {
  return TemperatureLadder::geometric(n, T_min, T_max);
}

} // namespace mcmc
