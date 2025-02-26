#pragma once
#include <cstddef>
#if defined(__CUDACC__) && !defined(__CUDA__)
#define __CUDA__
#endif
#include "OscillationParameters.h"
#include "constants.h"
#include "propagator.hpp"
#include <memory>

#ifdef __CUDA__
#include <cuda/std/mdspan>
#include <cuda/std/span>
#endif

class ParProb3ppOscillation {
public:
  ParProb3ppOscillation(const std::vector<oscillaton_calc_precision> &Ebin,
                        const std::vector<oscillaton_calc_precision> &costhbin);
  ~ParProb3ppOscillation();
  ParProb3ppOscillation(const ParProb3ppOscillation &);
  ParProb3ppOscillation(ParProb3ppOscillation &&) noexcept = default;
  ParProb3ppOscillation &operator=(const ParProb3ppOscillation &);
  ParProb3ppOscillation &operator=(ParProb3ppOscillation &&) noexcept = default;

  // void proposeStep();

  [[nodiscard]] std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta,
          const OscillationParameters &p) const;

  ///> The 3D probability histogram
  ///> [0-neutrino, 1-antineutrino][from: 0-nue, 1-nuebar][to: 0-nue, 1-nuebar]
  [[nodiscard]] std::array<std::array<std::array<TH2D, 2>, 2>, 2>
  GetProb_Hists(const std::vector<double> &Ebin,
                const std::vector<double> &costhbin,
                const OscillationParameters &p);

  ///> The 3D probability histogram
  ///> [0-neutrino, 1-antineutrino][from: 0-nue, 1-numu, 2-nutau][to: 0-nue,
  /// 1-numu, 2-nutau]
  [[nodiscard]] std::array<std::array<std::array<TH2D, 3>, 3>, 2>
  GetProb_Hists_3F(const std::vector<double> &Ebin,
                   const std::vector<double> &costhbin,
                   const OscillationParameters &p);

  void re_calculate(const OscillationParameters &p);

#ifdef __CUDA__

  using oscillaton_span_t = cuda::std::mdspan<
      oscillaton_calc_precision,
      cuda::std::extents<size_t, 3, 3, cuda::std::dynamic_extent,
                         cuda::std::dynamic_extent>>;
  oscillaton_span_t get_dev_span_neutrino();
  oscillaton_span_t get_dev_span_antineutrino();
#endif

private:
  std::shared_ptr<cudaprob3::Propagator<oscillaton_calc_precision>>
      propagator_neutrino;
  std::shared_ptr<cudaprob3::Propagator<oscillaton_calc_precision>>
      propagator_antineutrino;
  void load_state(cudaprob3::Propagator<oscillaton_calc_precision> &to_load,
                  const OscillationParameters &p, bool init = false);
  void load_state(cudaprob3::Propagator<oscillaton_calc_precision> &to_load,
                  bool init = true);
  std::vector<oscillaton_calc_precision> Ebins, costheta_bins;
};