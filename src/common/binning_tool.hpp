#pragma once

#include <stdexcept>
#include <vector>

template <class T> std::vector<T> linspace(T Emin, T Emax, unsigned int div) {
  if (div == 0)
    throw std::length_error("div == 0");

  std::vector<T> linpoints(div, 0.0);

  T step_lin = (Emax - Emin) / T(div - 1);

  T EE = Emin;

  for (unsigned int i = 0; i < div - 1; i++, EE += step_lin)
    linpoints[i] = EE;

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

  logpoints[0] = Emin;
  T EE = Emin_log + step_log;
  for (unsigned int i = 1; i < div - 1; i++, EE += step_log)
    logpoints[i] = exp(EE);
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