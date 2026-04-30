#pragma once
#include <cstddef>
#include "IHistogramPropagator.h"
#include "OscillationParameters.h"
#include "constants.h"
#include <memory>
#include <vector>

#if defined(__CUDACC__)
#include <cuda/std/mdspan>
#endif

namespace cudaprob3 {
template<typename T> class SingleGPUCalculator;
}

class ParProb3ppOscillation : public IHistogramPropagator {
public:
  ParProb3ppOscillation(const std::vector<oscillaton_calc_precision> &Ebin,
                        const std::vector<oscillaton_calc_precision> &costhbin);
  ~ParProb3ppOscillation();
  ParProb3ppOscillation(const ParProb3ppOscillation &);
  ParProb3ppOscillation(ParProb3ppOscillation &&) noexcept = default;
  ParProb3ppOscillation &operator=(const ParProb3ppOscillation &);
  ParProb3ppOscillation &operator=(ParProb3ppOscillation &&) noexcept = default;

  [[nodiscard]] std::array<std::array<double, 3>, 3>
  GetProb(int flavor, double E, double costheta,
          const OscillationParameters &p) const;

  [[nodiscard]] std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2>
  GetProb_Hists_POD(const std::vector<double> &Ebin,
                    const std::vector<double> &costhbin,
                    const OscillationParameters &p) override;

  [[nodiscard]] std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2>
  GetProb_Hists_3F_POD(const std::vector<double> &Ebin,
                       const std::vector<double> &costhbin,
                       const OscillationParameters &p) override;

  void re_calculate(const OscillationParameters &p) override;

#if defined(__CUDACC__)
  // Result span uses oscillaton_calc_precision, matching the full pipeline precision.
  using oscillaton_span_t = cuda::std::mdspan<
      const oscillaton_calc_precision,
      cuda::std::extents<size_t, 3, 3, cuda::std::dynamic_extent,
                         cuda::std::dynamic_extent>>;
  oscillaton_span_t get_dev_span_neutrino();
  oscillaton_span_t get_dev_span_antineutrino();
#endif

private:
  std::shared_ptr<cudaprob3::SingleGPUCalculator<oscillaton_calc_precision>> calculator_neutrino_;
  std::shared_ptr<cudaprob3::SingleGPUCalculator<oscillaton_calc_precision>> calculator_antineutrino_;
  std::vector<oscillaton_calc_precision> Ebins, costheta_bins;
};
