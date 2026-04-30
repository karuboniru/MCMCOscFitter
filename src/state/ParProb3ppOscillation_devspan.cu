#include "ParProb3ppOscillation.h"
#include "../../external/CUDAProb3/src/calculators/single_gpu_calculator.cuh"
#include "../../external/CUDAProb3/include/cudaprob3/oscillation_params.hpp"

#include <cmath>

ParProb3ppOscillation::oscillaton_span_t
ParProb3ppOscillation::get_dev_span_neutrino() {
  return oscillaton_span_t(calculator_neutrino_->getDeviceResultPtr(),
                            static_cast<size_t>(costheta_bins.size()),
                            static_cast<size_t>(Ebins.size()));
}

ParProb3ppOscillation::oscillaton_span_t
ParProb3ppOscillation::get_dev_span_antineutrino() {
  return oscillaton_span_t(calculator_antineutrino_->getDeviceResultPtr(),
                            static_cast<size_t>(costheta_bins.size()),
                            static_cast<size_t>(Ebins.size()));
}

void ParProb3ppOscillation::re_calculate_device(const OscillationParameters &p) {
  cudaprob3::OscillationParams params(
      std::asin(std::sqrt(p.GetT12())),
      std::asin(std::sqrt(p.GetT13())),
      std::asin(std::sqrt(p.GetT23())),
      p.GetDeltaCP(),
      p.GetDM21sq(), p.GetDM32sq());

  calculator_neutrino_->calculateDeviceOnly(params, cudaprob3::NeutrinoType::Neutrino);
  calculator_antineutrino_->calculateDeviceOnly(params, cudaprob3::NeutrinoType::Antineutrino);
}

cudaStream_t ParProb3ppOscillation::getComputeStream() const noexcept {
  return calculator_neutrino_->getComputeStream();
}
