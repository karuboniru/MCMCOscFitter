#pragma once

#include <array>
#include <cmath>
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
std::vector<T> to_center_g(const std::vector<U> &vec) {
  std::vector<T> ret(vec.size() - 1);
  for (size_t i = 0; i < vec.size() - 1; i++) {
    ret[i] = sqrt(vec[i] * vec[i + 1]);
  }
  return ret;
}

template <class T, typename U = T>
std::vector<T> divide_bins_log(const std::vector<U> &vec, size_t multiplier) {
  std::vector<T> ret((vec.size() - 1) * multiplier + 1);
  for (size_t i = 0; i < vec.size() - 1; i++) {
    auto step = (log(vec[i + 1]) - log(vec[i])) / (multiplier);
    for (size_t j = 0; j < multiplier + 1; j++) {
      ret[i * multiplier + j] = exp(log(vec[i]) + step * j);
    }
  }
  return ret;
}

template <class T, typename U = T>
std::vector<T> divide_bins(const std::vector<U> &vec, size_t multiplier) {
  std::vector<T> ret((vec.size() - 1) * multiplier + 1);
  for (size_t i = 0; i < vec.size() - 1; i++) {
    auto step = (vec[i + 1] - vec[i]) / (multiplier);
    for (size_t j = 0; j < multiplier + 1; j++) {
      ret[i * multiplier + j] = vec[i] + step * j;
    }
  }
  return ret;
}

// template <class T, typename U = T>
// std::vector<T> to_center(const std::vector<U> &vec, size_t multiplier) {
//   std::vector<T> ret((vec.size() - 1) * multiplier);
//   for (size_t i = 0; i < vec.size() - 1; i++) {
//     auto step = (vec[i + 1] - vec[i]) / (multiplier + 1);
//     for (size_t j = 0; j < multiplier; j++) {
//       ret[i * multiplier + j] = vec[i] + step * (j + 1);
//     }
//   }
//   return ret;
// }
template <class T>
std::vector<std::array<std::pair<size_t, T>, 2>>
build_rebin_map(const std::vector<T> &fine_bins,
                const std::vector<T> &new_bins) {
  // std::vector<std::pair<size_t, double>> ret;
  std::vector<std::array<std::pair<size_t, T>, 2>> ret;
  for (size_t fine_bin_index = 0, current_new_bin_id = 0;
       fine_bin_index < fine_bins.size() - 1; ++fine_bin_index) {
    auto fine_bin_lower = fine_bins[fine_bin_index];
    auto fine_bin_upper = fine_bins[fine_bin_index + 1];

    auto new_bin_lower = new_bins[current_new_bin_id];
    auto new_bin_upper = new_bins[current_new_bin_id + 1];

    if (fine_bin_lower >= new_bin_lower && fine_bin_upper <= new_bin_upper) {
      // the fine bin is fully contained in the new bin
      ret.push_back(
          std::to_array({std::pair<size_t, double>{current_new_bin_id, 1.0},
                         std::pair<size_t, double>{0, 0}}));
    } else if (fine_bin_lower < new_bin_lower) {
      // the fine bin sits on the lower edge of the new bin
      auto fraction =
          (new_bin_lower - fine_bin_lower) / (fine_bin_upper - fine_bin_lower);
      ret.push_back(std::to_array(
          {std::pair<size_t, double>{current_new_bin_id, fraction},
           std::pair<size_t, double>{current_new_bin_id - 1, 1 - fraction}}));
    } else if (fine_bin_upper > new_bin_upper) {
      // the fine bin sits on the upper edge of the new bin
      auto fraction =
          (fine_bin_upper - new_bin_upper) / (fine_bin_upper - fine_bin_lower);
      ret.push_back(std::to_array(
          {std::pair<size_t, double>{current_new_bin_id, 1 - fraction},
           std::pair<size_t, double>{current_new_bin_id + 1, fraction}}));
      current_new_bin_id++;
    } else {
      // the fine bin is fully outside the new bin
      // should not happen
      throw std::runtime_error("fine bin is fully outside the new bin");
      // ret.push_back(std::to_array(
      //     {std::pair<size_t, double>{0, 0}, std::pair<size_t, double>{0,
      //     0}}));
    }
  }
  return ret;
}