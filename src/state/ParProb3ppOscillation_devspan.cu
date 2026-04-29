#include "ParProb3ppOscillation.h"
#include "../../external/CUDAProb3/src/calculators/single_gpu_calculator.cuh"

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
