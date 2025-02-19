#pragma once
#include <cstddef>
#if defined(__CUDACC__) && !defined(__CUDA__)
#define __CUDA__
#endif
#include "OscillationParameters.h"
#include "constants.h"
#include <memory>

#ifdef __CUDA__
#include <cuda/std/mdspan>
#include <cuda/std/span>
#endif

namespace cudaprob3 {
#ifndef __CUDA__
template <class oscillaton_calc_precision_T> class CpuPropagator;
#else
template <class oscillaton_calc_precision_T> class CudaPropagatorSingle;
#endif
} // namespace cudaprob3
class ParProb3ppOscillation : public OscillationParameters {
public:
  ParProb3ppOscillation(const std::vector<oscillaton_calc_precision> &Ebin,
                        const std::vector<oscillaton_calc_precision> &costhbin);
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
  GetProb_Hists(const std::vector<double> &Ebin,
                const std::vector<double> &costhbin);

  ///> The 3D probability histogram
  ///> [0-neutrino, 1-antineutrino][from: 0-nue, 1-numu, 2-nutau][to: 0-nue,
  /// 1-numu, 2-nutau]
  [[nodiscard]] std::array<std::array<std::array<TH2D, 3>, 3>, 2>
  GetProb_Hists_3F(const std::vector<double> &Ebin,
                   const std::vector<double> &costhbin);

  void re_calculate() override;

#ifdef __CUDA__

  using oscillaton_span_t = cuda::std::mdspan<
      oscillaton_calc_precision,
      cuda::std::extents<size_t, 3, 3, cuda::std::dynamic_extent,
                         cuda::std::dynamic_extent>>;
  oscillaton_span_t get_dev_span_neutrino();
  oscillaton_span_t get_dev_span_antineutrino();
#endif

private:
#ifndef __CUDA__
  std::shared_ptr<cudaprob3::CpuPropagator<oscillaton_calc_precision>>
      propagator_neutrino;
  std::shared_ptr<cudaprob3::CpuPropagator<oscillaton_calc_precision>>
      propagator_antineutrino;
  void load_state(cudaprob3::CpuPropagator<oscillaton_calc_precision> &to_load,
                  bool init = false);
#else
  std::shared_ptr<cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>
      propagator_neutrino;
  std::shared_ptr<cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision>>
      propagator_antineutrino;
  void load_state(
      cudaprob3::CudaPropagatorSingle<oscillaton_calc_precision> &to_load,
      bool init = false);

#endif
  std::vector<oscillaton_calc_precision> Ebins, costheta_bins;
};