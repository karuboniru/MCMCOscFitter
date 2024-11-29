#pragma once

#include <stdexcept>
#include <vector>

template <class T> std::vector<T> linspace(T Emin, T Emax, unsigned int div) {
  if (div == 0)
    throw std::length_error("div == 0");

  std::vector<T> linpoints(div, 0.0);

  T step_lin = (Emax - Emin) / T(div - 1);

  for (unsigned int i = 0; i < div - 1; i++) {
    linpoints[i] = Emin + (step_lin * i);
  }

  linpoints[div - 1] = Emax;

  return linpoints;
}

template <class T> std::vector<T> logspace(T Emin, T Emax, unsigned int div) {
  if (div == 0)
    throw std::length_error("div == 0");
  std::vector<T> logpoints(div, 0.0);

  T Emin_log, Emax_log;
  Emin_log = log(Emin);
  Emax_log = log(Emax);

  T step_log = (Emax_log - Emin_log) / T(div - 1);

  for (unsigned int i = 0; i < div - 1; i++) {
    logpoints[i] = exp(Emin_log + (step_log * i));
  }
  logpoints[div - 1] = Emax;
  return logpoints;
}

template <class T, typename U = T>
std::vector<T> to_center(const std::vector<U> &vec) {
  std::vector<T> ret(vec.size() - 1);
  for (size_t i = 0; i < vec.size() - 1; i++) {
    ret[i] = (vec[i] + vec[i + 1]) / 2;
  }
  return ret;
}

template <class T, typename U = T>
std::vector<T> to_center(const std::vector<U> &vec, size_t multiplier) {
  std::vector<T> ret((vec.size() - 1) * multiplier);
  for (size_t i = 0; i < vec.size() - 1; i++) {
    auto step = (vec[i + 1] - vec[i]) / (multiplier + 1);
    for (size_t j = 0; j < multiplier; j++) {
      ret[i * multiplier + j] = vec[i] + step * (j + 1);
    }
  }
  return ret;
}