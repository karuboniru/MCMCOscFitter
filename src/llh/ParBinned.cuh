#include "OscillationParameters.h"
#include "ParBinnedInterface.h"
#include <ParProb3ppOscillation.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

using propgator_type = ParProb3ppOscillation;
TH2D vec_to_hist(const thrust::host_vector<oscillaton_calc_precision> &from_vec,
                 size_t costh_bins, size_t e_bins);
TH2D vec_to_hist(const thrust::host_vector<oscillaton_calc_precision> &from_vec,
                 const std::vector<double> &costh_bins_v,
                 const std::vector<double> &e_bins_v);
class ParBinned : public OscillationParameters, public ModelDataLLH {
public:
  ParBinned(std::vector<double> Ebins, std::vector<double> costheta_bins,
            double scale_ = 1., size_t E_rebin_factor = 1,
            size_t costh_rebin_factor = 1, double IH_Bias = 1.0);

  ParBinned(const ParBinned &) = default;
  ParBinned(ParBinned &&) noexcept = default;
  ParBinned &operator=(const ParBinned &) = default;
  ParBinned &operator=(ParBinned &&) noexcept = default;
  ~ParBinned() override = default;

  void proposeStep() final;

  // virtual double GetLogLikelihood() const override;
  [[nodiscard]] double
  GetLogLikelihoodAgainstData(const StateI &dataset) const final;

  [[nodiscard]] SimpleDataHist GenerateData() const;
  [[nodiscard]] SimpleDataHist GenerateData_NoOsc() const;

  void Print() const {
    // flux_hist_numu.Print();
    // xsec_hist_numu.Print();
  }

  void flip_hierarchy();

  void Save_prob_hist(const std::string &name);

  [[nodiscard]] double GetLogLikelihood() const final;

  void UpdatePrediction();

  void SaveAs(const char *filename) const;

  auto vec2span_fine(auto &vec) const {
    return cuda::std::mdspan<
        std::remove_reference_t<decltype(*vec.data().get())>,
        cuda::std::extents<size_t, cuda::std::dynamic_extent,
                           cuda::std::dynamic_extent>>(
        vec.data().get(), costh_fine_bin_count, E_fine_bin_count);
  }

  auto vec2span_analysis(auto &vec) const {
    return cuda::std::mdspan<
        std::remove_reference_t<decltype(*vec.data().get())>,
        cuda::std::extents<size_t, cuda::std::dynamic_extent,
                           cuda::std::dynamic_extent>>(
        vec.data().get(), costh_analysis_bin_count, E_analysis_bin_count);
  }

private:
  std::shared_ptr<propgator_type> propagator;

  std::vector<double> Ebins, costheta_bins;
  std::vector<double> Ebins_analysis, costheta_analysis;

  size_t E_rebin_factor;
  size_t costh_rebin_factor;

  size_t E_fine_bin_count, costh_fine_bin_count, E_analysis_bin_count,
      costh_analysis_bin_count;

  // index: [cosine, energy]
  thrust::device_vector<oscillaton_calc_precision> Prediction_hist_numu,
      Prediction_hist_numubar, Prediction_hist_nue, Prediction_hist_nuebar;

  auto vec2hist_analysis(const auto &vec) const {
    return vec_to_hist(vec, costheta_analysis, Ebins_analysis);
  }

  double log_ih_bias;
};