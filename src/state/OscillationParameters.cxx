#include "OscillationParameters.h"

#include <TRandom.h>
#include <cmath>
#include <cstdlib>

[[gnu::const]]
double log_llh_gaussian(double x1, double x2, double sigma) {
  // return exp(-0.5 * pow((x1 - x2) / sigma, 2)) / (sqrt(2 * M_1_PI) * sigma);
  return -0.5 * pow((x1 - x2) / sigma, 2);
  // return -0.5 * pow((x1 - x2) / sigma, 2) - log(2 * M_1_PI) / 2 - log(sigma);
}

void OscillationParameters::proposeStep() {
  const bool flip = gRandom->Rndm() < 0.8;
  // const bool use_NH = flip ^ (NH_DM2 < 0);
  constexpr double distance_factor = 0.1;
  is_NH = flip ^ is_NH;
  auto &current_DM2 = is_NH ? NH_DM2 : IH_DM2;
  auto &current_T23 = is_NH ? NH_T23 : IH_T23;
  auto &current_T13 = is_NH ? NH_T13 : IH_T13;
  auto &current_Dm2 = is_NH ? NH_Dm2 : IH_Dm2;
  auto &current_T12 = is_NH ? NH_T12 : IH_T12;
  auto &current_DCP = is_NH ? NH_DCP : IH_DCP;

  // NH_DM2 = gRandom->Gaus(NH_DM2, 0.1);

  current_DCP = gRandom->Gaus(current_DCP, 0.1);
  if (current_DCP > M_PI) {
    current_DCP -= 2 * M_PI;
  } else if (current_DCP < -M_PI) {
    current_DCP += 2 * M_PI;
  }

  current_DM2 = gRandom->Gaus(current_DM2, sigma_DM2 * distance_factor);
  current_Dm2 = gRandom->Gaus(current_Dm2, sigma_dm2 * distance_factor);
  current_T23 = gRandom->Gaus(current_T23, sigma_t23 * distance_factor);
  current_T13 = gRandom->Gaus(current_T13, sigma_t13 * distance_factor);
  current_T12 = gRandom->Gaus(current_T12, sigma_t12 * distance_factor);

  // NH_DM2 = use_NH ? NH_DM2 : -NH_DM2;
  // NH_Dm2 = fabs(NH_Dm2);
  // NH_T23 = fabs(NH_T23);
  // NH_T13 = fabs(NH_T13);
  // NH_T12 = fabs(NH_T12);
}

double OscillationParameters::GetLogLikelihood() const {
  // return 0;
  if (is_NH) {
    return log_llh_gaussian(GetDM2(), DM2, sigma_DM2) +
           log_llh_gaussian(GetT23(), Theta23, sigma_t23) +
           log_llh_gaussian(GetT13(), Theta13, sigma_t13) +
           log_llh_gaussian(GetDm2(), dm2, sigma_dm2) +
           log_llh_gaussian(GetT12(), Theta12, sigma_t12);
  }
  return log_llh_gaussian(GetDM2(), DM2_IH, sigma_DM2_IH) +
         log_llh_gaussian(GetT23(), Theta23_IH, sigma_t23_IH) +
         log_llh_gaussian(GetT13(), Theta13, sigma_t13) +
         log_llh_gaussian(GetDm2(), dm2, sigma_dm2) +
         log_llh_gaussian(GetT12(), Theta12, sigma_t12);
}
