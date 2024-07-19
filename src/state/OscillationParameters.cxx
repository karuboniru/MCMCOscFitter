#include "OscillationParameters.h"
#include "BargerPropagator.h"

#include <TRandom.h>
#include <cmath>
#include <cstdlib>

double log_llh_gaussian(double x1, double x2, double sigma) {
  // return exp(-0.5 * pow((x1 - x2) / sigma, 2)) / (sqrt(2 * M_1_PI) * sigma);
  return -0.5 * pow((x1 - x2) / sigma, 2) - log(2 * M_1_PI) / 2 - log(sigma);
}

void OscillationParameters::proposeStep() {
  const bool flip = gRandom->Rndm() < 0.1;
  const bool use_NH = flip && (current_DM2 < 0);
  constexpr double distance_factor = 0.1;

  current_DM2 = gRandom->Gaus(fabs(current_DM2), sigma_DM2 * distance_factor);
  // current_dm2 = gRandom->Gaus(current_dm2, sigma_dm2 * distance_factor);
  current_Theta23 = gRandom->Gaus(current_Theta23, sigma_t23 * distance_factor);
  // current_Theta13 = gRandom->Gaus(current_Theta13, sigma_t13 *
  // distance_factor); current_Theta12 = gRandom->Gaus(current_Theta12,
  // sigma_t12 * distance_factor);

  current_DM2 = use_NH ? current_DM2 : -current_DM2;
  // current_dm2 = fabs(current_dm2);
  current_Theta23 = fabs(current_Theta23);
  // current_Theta13 = fabs(current_Theta13);
  // current_Theta12 = fabs(current_Theta12);
}

double OscillationParameters::GetLogLikelihood() const {
  if (current_DM2 > 0) {
    return log_llh_gaussian(current_DM2, DM2, sigma_DM2) +
           log_llh_gaussian(current_Theta23, Theta23, sigma_t23) +
           log_llh_gaussian(current_Theta13, Theta13, sigma_t13) +
           log_llh_gaussian(current_dm2, dm2, sigma_dm2) +
           log_llh_gaussian(current_Theta12, Theta12, sigma_t12);
  } else {
    return log_llh_gaussian(current_DM2, DM2_IH, sigma_DM2_IH) +
           log_llh_gaussian(current_Theta23, Theta23_IH, sigma_t23_IH) +
           log_llh_gaussian(current_Theta13, Theta13, sigma_t13) +
           log_llh_gaussian(current_dm2, dm2, sigma_dm2) +
           log_llh_gaussian(current_Theta12, Theta12, sigma_t12);
  }
}

std::array<std::array<double, 3>, 3>
Prob3ppOscillation::GetProb(int flavor, double E, double costheta) const {
  flavor = flavor / abs(flavor);
  BargerPropagator b;
  b.SetMNS(current_Theta12, current_Theta13, current_Theta23, current_dm2,
           current_DM2, 0. /*delta cp*/, E, true, flavor);
  b.DefinePath(costheta, 25);
  b.propagate(flavor);
  std::array<std::array<double, 3>, 3> ret{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ret[i][j] = b.GetProb(flavor * (i + 1), flavor * (j + 1));
    }
  }
  return ret;
}