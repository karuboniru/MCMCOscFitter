#pragma once

#include "OscillationParameters.h"
#include <memory>

namespace cudaprob3 {
#ifndef __CUDA__
template <class FLOAT_T> class CpuPropagator;
#else
template <class FLOAT_T> class CudaPropagatorSingle;
#endif
} // namespace cudaprob3
class ParProb3ppOscillation : public OscillationParameters {
public:
  ParProb3ppOscillation(const std::vector<float> &Ebin,
                        const std::vector<float> &costhbin);
  ~ParProb3ppOscillation() override;
  ParProb3ppOscillation(const ParProb3ppOscillation &);
  ParProb3ppOscillation(ParProb3ppOscillation &&) noexcept = default;
  ParProb3ppOscillation &operator=(const ParProb3ppOscillation &);
  ParProb3ppOscillation &operator=(ParProb3ppOscillation &&) noexcept = default;

  void proposeStep() override;

  [[nodiscard]] std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta) const override;

  ///> The 3D probability histogram
  ///> [0-neutrino, 1-antineutrino][from: 0-nue, 1-nuebar][to: 0-nue, 1-nuebar]
  [[nodiscard]] std::array<std::array<std::array<TH2D, 2>, 2>, 2>
  GetProb_Hist(std::vector<double> Ebin, std::vector<double> costhbin);

private:
#ifndef __CUDA__
  std::unique_ptr<cudaprob3::CpuPropagator<float>> propagator_neutrino;
  std::unique_ptr<cudaprob3::CpuPropagator<float>> propagator_antineutrino;
  void load_state(cudaprob3::CpuPropagator<float> &to_load);
#else
  std::unique_ptr<cudaprob3::CudaPropagatorSingle<float>> propagator_neutrino;
  std::unique_ptr<cudaprob3::CudaPropagatorSingle<float>>
      propagator_antineutrino;
  void load_state(cudaprob3::CudaPropagatorSingle<float> &to_load);

#endif
  std::vector<float> Ebins, costheta_bins;
};