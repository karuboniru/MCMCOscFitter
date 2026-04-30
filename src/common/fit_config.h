#pragma once

#include "constants.h"
#include <cstddef>

/// Centralised configuration values for all executables.
/// Previously these were hardcoded as magic numbers in every main().
struct FitConfig {
  // Binning — bin edges are linspace/logspace with n+1 points.
  static constexpr size_t n_costheta_bins     = 480;
  static constexpr size_t n_energy_bins       = 400;
  static constexpr double e_min               = 0.1;
  static constexpr double e_max               = 20.0;

  // Rebinning factors (fine → analysis bins).
  static constexpr size_t E_rebin_factor      = 40;
  static constexpr size_t costh_rebin_factor  = 40;

  // Physics scale and bias.
  static constexpr double scale_factor        = scale_factor_6y;
  static constexpr double ih_bias             = 8000.;

  // MCMC proposal step size (was hardcoded 0.1 in OscillationParameters).
  static constexpr double proposal_distance   = 0.1;
};
