#include "OscillationParameters.h"

#include <TRandom.h>
#include <cmath>
#include <cstdlib>

[[gnu::const]]
double log_llh_gaussian(double x1, double x2, double sigma) {
  return -0.5 * pow((x1 - x2) / sigma, 2);
}

[[gnu::const]]
double log_llh_gaussian_cyd(double x1, double x2, double sigma) {
  auto diff = std::abs(x1 - x2);
  // for delta cp we need to wrap around
  if (diff > M_PI) {
    diff = 2 * M_PI - diff;
  }
  return -0.5 * pow(diff / sigma, 2);
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
  return GetLogLikelihood(current_toggle);
}

double
OscillationParameters::GetLogLikelihood(const pull_toggle &toggle) const {
  auto sigma_DM2_H = is_NH ? sigma_DM2 : sigma_DM2_IH;
  auto sigma_t23_H = is_NH ? sigma_t23 : sigma_t23_IH;
  auto DM2_H = is_NH ? DM2 : DM2_IH;
  auto Theta23_H = is_NH ? Theta23 : Theta23_IH;
  double llh{};
  if (toggle[0]) {
    llh += log_llh_gaussian(GetDM32sq(), DM2_H, sigma_DM2_H);
  }
  if (toggle[1]) {
    llh += log_llh_gaussian(GetDM21sq(), dm2, sigma_dm2);
  }
  if (toggle[2]) {
    llh += log_llh_gaussian(GetT23(), Theta23_H, sigma_t23_H);
  }
  if (toggle[3]) {
    llh += log_llh_gaussian(GetT13(), Theta13, sigma_t13);
  }
  if (toggle[4]) {
    llh += log_llh_gaussian(GetT12(), Theta12, sigma_t12);
  }
  if (toggle[5]) {
    llh += log_llh_gaussian_cyd(GetDeltaCP(), DCP, sigma_DCP);
  }
  return llh;
}

void OscillationParameters::set_param(const param &p) {
  if (p.DM2 > 0) {
    NH_DM2 = p.DM2;
    NH_T23 = p.T23;
    NH_T13 = p.T13;
    NH_Dm2 = p.Dm2;
    NH_T12 = p.T12;
    NH_DCP = p.DCP;
    is_NH = true;
  } else {
    IH_DM2 = p.DM2;
    IH_T23 = p.T23;
    IH_T13 = p.T13;
    IH_Dm2 = p.Dm2;
    IH_T12 = p.T12;
    IH_DCP = p.DCP;
    is_NH = false;
  }
  // re_calculate();
}

const std::array<std::string,6> pull_toggle::names{"DM32", "DM21", "T23", "T13", "T12", "DCP"};