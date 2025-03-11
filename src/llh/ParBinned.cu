#include "ParProb3ppOscillation.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "genie_xsec.h"
// #include "hondaflux2d.h"
#include "WingFlux.h"

#include <SimpleDataHist.h>
#include <TF3.h>
#include <TH2.h>
#include <cmath>
#include <print>
#include <ranges>
#include <type_traits>
#include <utility>

#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include "ParProb3ppOscillation.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataHist.h"
#include "constants.h"
#include "genie_xsec.h"
#include <format>
#include <functional>

#include "OscillationParameters.h"
#include "ParBinned.cuh"
#include "ParBinnedKernels.cuh"

#define CUERR                                                                  \
  {                                                                            \
    cudaError_t err;                                                           \
    if ((err = cudaGetLastError()) != cudaSuccess) {                           \
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "          \
                << __FILE__ << ", line " << __LINE__ << std::endl;             \
      exit(1);                                                                 \
    }                                                                          \
  }

using span_2d_hist_t =
    cuda::std::mdspan<oscillaton_calc_precision,
                      cuda::std::extents<size_t, cuda::std::dynamic_extent,
                                         cuda::std::dynamic_extent>>;
using const_span_2d_hist_t =
    cuda::std::mdspan<const oscillaton_calc_precision,
                      cuda::std::extents<size_t, cuda::std::dynamic_extent,
                                         cuda::std::dynamic_extent>>;

thrust::host_vector<oscillaton_calc_precision> TH1_to_hist(const TH1D &hist) {
  thrust::host_vector<oscillaton_calc_precision> ret(hist.GetNbinsX());
  for (int i = 0; i < hist.GetNbinsX(); i++) {
    ret[i] = hist.GetBinContent(i + 1);
  }
  return ret;
}
thrust::host_vector<oscillaton_calc_precision> TH2D_to_hist(const TH2D &hist) {
  thrust::host_vector<oscillaton_calc_precision> ret(hist.GetNbinsX() *
                                                     hist.GetNbinsY());
  span_2d_hist_t ret_span(
      ret.data(), hist.GetNbinsY(),
      hist.GetNbinsX()); // notice in vector, its costh then E
  // but in TH2D, its E then costh
  for (int x = 1; x <= hist.GetNbinsX(); x++) {
    for (int y = 1; y <= hist.GetNbinsY(); y++) {
#if __cpp_multidimensional_subscript
      ret_span[y - 1, x - 1] = hist.GetBinContent(x, y);
#else
      ret_span(y - 1, x - 1) = hist.GetBinContent(x, y);
#endif
    }
  }
  return ret;
}

TH2D vec_to_hist(const thrust::host_vector<oscillaton_calc_precision> &from_vec,
                 size_t costh_bins, size_t e_bins) {
  TH2D ret("", "", e_bins, 0, e_bins, costh_bins, 0, costh_bins);
  const_span_2d_hist_t ret_span(from_vec.data(), costh_bins, e_bins);
  for (int x = 1; x <= e_bins; x++) {
    for (int y = 1; y <= costh_bins; y++) {
#if __cpp_multidimensional_subscript
      ret.SetBinContent(x, y, ret_span[y - 1, x - 1]);
#else
      ret.SetBinContent(x, y, ret_span(y - 1, x - 1));
#endif
    }
  }
  return ret;
}

TH2D vec_to_hist(const thrust::host_vector<oscillaton_calc_precision> &from_vec,
                 const std::vector<double> &costh_bins_v,
                 const std::vector<double> &e_bins_v) {
  auto e_bins = e_bins_v.size() - 1;
  auto costh_bins = costh_bins_v.size() - 1;
  if (from_vec.size() != e_bins * costh_bins) {
    throw std::runtime_error("from_vec size does not match the number of bins");
  }
  TH2D ret("", "", e_bins, e_bins_v.data(), costh_bins, costh_bins_v.data());
  const_span_2d_hist_t ret_span(from_vec.data(), costh_bins, e_bins);
  for (int x = 1; x <= e_bins; x++) {
    for (int y = 1; y <= costh_bins; y++) {
#if __cpp_multidimensional_subscript
      ret.SetBinContent(x, y, ret_span[y - 1, x - 1]);
#else
      ret.SetBinContent(x, y, ret_span(y - 1, x - 1));
#endif
    }
  }
  return ret;
}

auto vec2span_1d(auto &vec) {
  return cuda::std::span<std::remove_reference_t<decltype(*vec.data().get())>>(
      vec.data().get(), vec.size());
}

// xsec and flux can be considered
// as constant for the lifetime of the object
// so we can use a global device input instance
class global_device_input_instance {
public:
  static global_device_input_instance &
  get_instance(const std::vector<double> &Ebins = {},
               const std::vector<double> &costheta_bins = {},
               double scale_ = 1.) {
    static global_device_input_instance instance{Ebins, costheta_bins, scale_};
    return instance;
  }

private:
  [[nodiscard]] auto vec2span_fine(auto &vec) const {
    return cuda::std::mdspan<
        std::remove_reference_t<decltype(*vec.data().get())>,
        cuda::std::extents<size_t, cuda::std::dynamic_extent,
                           cuda::std::dynamic_extent>>(
        vec.data().get(), costh_fine_bin_count, E_fine_bin_count);
  }

public:
  [[nodiscard]] auto get_flux_numu() const {
    return vec2span_fine(flux_hist_numu);
  }
  [[nodiscard]] auto get_flux_numubar() const {
    return vec2span_fine(flux_hist_numubar);
  }
  [[nodiscard]] auto get_flux_nue() const {
    return vec2span_fine(flux_hist_nue);
  }
  [[nodiscard]] auto get_flux_nuebar() const {
    return vec2span_fine(flux_hist_nuebar);
  }

  [[nodiscard]] auto get_xsec_numu() const {
    return vec2span_1d(xsec_hist_numu);
  }
  [[nodiscard]] auto get_xsec_numubar() const {
    return vec2span_1d(xsec_hist_numubar);
  }
  [[nodiscard]] auto get_xsec_nue() const { return vec2span_1d(xsec_hist_nue); }
  [[nodiscard]] auto get_xsec_nuebar() const {
    return vec2span_1d(xsec_hist_nuebar);
  }

  ~global_device_input_instance() = default;
  global_device_input_instance(const global_device_input_instance &) = delete;
  global_device_input_instance(global_device_input_instance &&) = delete;
  global_device_input_instance &
  operator=(const global_device_input_instance &) = delete;
  global_device_input_instance &
  operator=(global_device_input_instance &&) = delete;

private:
  global_device_input_instance(const std::vector<double> &Ebins,
                               const std::vector<double> &costheta_bins,
                               double scale_ = 1.)
      : // flux_hist_numu(TH2D_to_hist(
        //       flux_input.GetFlux_Hist(Ebins, costheta_bins, 14) * scale_)),
        //   flux_hist_numubar(TH2D_to_hist(
        //       flux_input.GetFlux_Hist(Ebins, costheta_bins, -14) * scale_)),
        //   flux_hist_nue(TH2D_to_hist(
        //       flux_input.GetFlux_Hist(Ebins, costheta_bins, 12) * scale_)),
        //   flux_hist_nuebar(TH2D_to_hist(
        //       flux_input.GetFlux_Hist(Ebins, costheta_bins, -12) * scale_)),
        flux_hist_numu(TH2D_to_hist(wingflux.GetFlux_Hist(14) * scale_)),
        flux_hist_numubar(TH2D_to_hist(wingflux.GetFlux_Hist(-14) * scale_)),
        flux_hist_nue(TH2D_to_hist(wingflux.GetFlux_Hist(12) * scale_)),
        flux_hist_nuebar(TH2D_to_hist(wingflux.GetFlux_Hist(-12) * scale_)),
        xsec_hist_numu(TH1_to_hist(xsec_input.GetXsecHistMixture(
            Ebins, 14, {{1000060120, 1.0}, {2212, H_to_C}}))),
        xsec_hist_numubar(TH1_to_hist(xsec_input.GetXsecHistMixture(
            Ebins, -14, {{1000060120, 1.0}, {2212, H_to_C}}))),
        xsec_hist_nue(TH1_to_hist(xsec_input.GetXsecHistMixture(
            Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}}))),
        xsec_hist_nuebar(TH1_to_hist(xsec_input.GetXsecHistMixture(
            Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}}))),
        E_fine_bin_count(Ebins.size() - 1),
        costh_fine_bin_count(costheta_bins.size() - 1) {
  }

  // index: [cosine, energy]
  thrust::device_vector<oscillaton_calc_precision> flux_hist_numu,
      flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;

  // 1D in energy
  thrust::device_vector<oscillaton_calc_precision> xsec_hist_numu,
      xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  size_t E_fine_bin_count, costh_fine_bin_count;
};

constexpr size_t warp_size = 32;

template <class T>
std::vector<T> stride(const std::vector<T> &vec, size_t stride) {
  std::vector<T> ret;
  ret.reserve(vec.size() / stride);
  for (size_t i = 0; i < vec.size(); i += stride) {
    ret.push_back(vec[i]);
  }
  return ret;
}

ParBinned::ParBinned(std::vector<double> Ebins_,
                     std::vector<double> costheta_bins_, double scale_,
                     size_t E_rebin_factor_, size_t costh_rebin_factor_,
                     double IH_bias_)
    : ModelDataLLH(),
      propagator{std::make_shared<ParProb3ppOscillation>(
          to_center<oscillaton_calc_precision>(Ebins_),
          to_center<oscillaton_calc_precision>(costheta_bins_))},
      Ebins(std::move(Ebins_)), costheta_bins(std::move(costheta_bins_)),
      Ebins_analysis(stride(Ebins, E_rebin_factor_)
                     // Ebins | std::views::stride(E_rebin_factor_) |
                     //              std::ranges::to<std::vector>()
                     ),
      costheta_analysis(
          stride(costheta_bins, costh_rebin_factor_)
          // costheta_bins |
          //                 std::views::stride(costh_rebin_factor_)
          //                 | std::ranges::to<std::vector>()
          ),
      E_rebin_factor(E_rebin_factor_), costh_rebin_factor(costh_rebin_factor_),
      E_fine_bin_count(Ebins.size() - 1),
      costh_fine_bin_count(costheta_bins.size() - 1),
      E_analysis_bin_count(E_fine_bin_count / E_rebin_factor),
      costh_analysis_bin_count(costh_fine_bin_count / costh_rebin_factor),
      Prediction_hist_numu(E_analysis_bin_count * costh_analysis_bin_count),
      Prediction_hist_numubar(E_analysis_bin_count * costh_analysis_bin_count),
      Prediction_hist_nue(E_analysis_bin_count * costh_analysis_bin_count),
      Prediction_hist_nuebar(E_analysis_bin_count * costh_analysis_bin_count),
      log_ih_bias(std::log(IH_bias_)) {
  global_device_input_instance::get_instance(Ebins, costheta_bins, scale_);
  UpdatePrediction();
}

void ParBinned::UpdatePrediction() {
  propagator->re_calculate(*this);
  auto span_prob_neutrino = propagator->get_dev_span_neutrino();
  CUERR
  auto span_prob_antineutrino = propagator->get_dev_span_antineutrino();
  CUERR

  auto &flux_xsec_device_input = global_device_input_instance::get_instance();
  // #ifndef __clang__
  cudaMemsetAsync(Prediction_hist_numu.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_numu.size());
  cudaMemsetAsync(Prediction_hist_numubar.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_numubar.size());
  cudaMemsetAsync(Prediction_hist_nue.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_nue.size());
  cudaMemsetAsync(Prediction_hist_nuebar.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_nuebar.size());
  calc_event_count_atomic_add<<<
      cuda::ceil_div(E_fine_bin_count * costh_fine_bin_count, warp_size),
      warp_size>>>(span_prob_neutrino, span_prob_antineutrino,
                   flux_xsec_device_input.get_flux_numu(),
                   flux_xsec_device_input.get_flux_numubar(),
                   flux_xsec_device_input.get_flux_nue(),
                   flux_xsec_device_input.get_flux_nuebar(),
                   flux_xsec_device_input.get_xsec_numu(),
                   flux_xsec_device_input.get_xsec_numubar(),
                   flux_xsec_device_input.get_xsec_nue(),
                   flux_xsec_device_input.get_xsec_nuebar(),
                   vec2span_analysis(Prediction_hist_numu),
                   vec2span_analysis(Prediction_hist_numubar),
                   vec2span_analysis(Prediction_hist_nue),
                   vec2span_analysis(Prediction_hist_nuebar), E_rebin_factor,
                   costh_rebin_factor);
}

void ParBinned::proposeStep() {
  OscillationParameters::proposeStep();
  UpdatePrediction();
}

namespace {
double TH2D_chi2(const TH2D &data, const TH2D &pred) {
  auto binsx = data.GetNbinsX();
  auto binsy = data.GetNbinsY();
  double chi2{};
  // #pragma omp parallel for reduction(+ : chi2) collapse(2)
  for (int x = 1; x <= binsx; x++) {
    for (int y = 1; y <= binsy; y++) {
      auto bin_data = data.GetBinContent(x, y);
      auto bin_pred = pred.GetBinContent(x, y);
      if (bin_data != 0) [[likely]]
        chi2 +=
            (bin_pred - bin_data) + bin_data * std::log(bin_data / bin_pred);
      else
        chi2 += bin_pred;
    }
  }
  return 2 * chi2;
}
} // namespace
double ParBinned::GetLogLikelihoodAgainstData(StateI const &dataset) const {
  auto casted = dynamic_cast<const SimpleDataHist &>(dataset);
  auto &data_hist_numu = casted.hist_numu;
  auto &data_hist_numubar = casted.hist_numubar;
  auto &data_hist_nue = casted.hist_nue;
  auto &data_hist_nuebar = casted.hist_nuebar;

  auto chi2_numu =
      TH2D_chi2(data_hist_numu, vec2hist_analysis(Prediction_hist_numu));
  auto chi2_numubar =
      TH2D_chi2(data_hist_numubar, vec2hist_analysis(Prediction_hist_numubar));
  auto chi2_nue =
      TH2D_chi2(data_hist_nue, vec2hist_analysis(Prediction_hist_nue));
  auto chi2_nuebar =
      TH2D_chi2(data_hist_nuebar, vec2hist_analysis(Prediction_hist_nuebar));

  auto llh = -0.5 * (chi2_numu + chi2_numubar + chi2_nue + chi2_nuebar);
  return llh;
}

SimpleDataHist ParBinned::GenerateData() const {
  SimpleDataHist data;
  data.hist_numu = vec2hist_analysis(Prediction_hist_numu);
  data.hist_numubar = vec2hist_analysis(Prediction_hist_numubar);
  data.hist_nue = vec2hist_analysis(Prediction_hist_nue);
  data.hist_nuebar = vec2hist_analysis(Prediction_hist_nuebar);
  return data;
}

SimpleDataHist ParBinned::GenerateData_NoOsc() const {
  auto &flux_xsec_device_input = global_device_input_instance::get_instance();

  SimpleDataHist data;
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_numu(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_numubar(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_nue(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_nuebar(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  calc_event_count_noosc_atomic<<<
      cuda::ceil_div(costh_fine_bin_count * E_fine_bin_count, warp_size),
      warp_size>>>(flux_xsec_device_input.get_flux_numu(),
                   flux_xsec_device_input.get_flux_numubar(),
                   flux_xsec_device_input.get_flux_nue(),
                   flux_xsec_device_input.get_flux_nuebar(),
                   flux_xsec_device_input.get_xsec_numu(),
                   flux_xsec_device_input.get_xsec_numubar(),
                   flux_xsec_device_input.get_xsec_nue(),
                   flux_xsec_device_input.get_xsec_nuebar(),
                   vec2span_analysis(no_osc_hist_numu),
                   vec2span_analysis(no_osc_hist_numubar),
                   vec2span_analysis(no_osc_hist_nue),
                   vec2span_analysis(no_osc_hist_nuebar), E_rebin_factor,
                   costh_rebin_factor);

  data.hist_numu = vec2hist_analysis(no_osc_hist_numu);
  data.hist_numubar = vec2hist_analysis(no_osc_hist_numubar);
  data.hist_nue = vec2hist_analysis(no_osc_hist_nue);
  data.hist_nuebar = vec2hist_analysis(no_osc_hist_nuebar);

  return data;
}

double ParBinned::GetLogLikelihood() const {
  auto llh = OscillationParameters::GetLogLikelihood();
  if (GetDM32sq() < 0) {
    llh += log_ih_bias;
  }
  return llh;
}

void ParBinned::SaveAs(const char *filename) const {
  // auto file = TFile::Open(filename, "RECREATE");
  // file->cd();

  // file->Add(flux_hist_numu.Clone("flux_hist_numu"));
  // file->Add(flux_hist_numubar.Clone("flux_hist_numubar"));
  // file->Add(flux_hist_nue.Clone("flux_hist_nue"));
  // file->Add(flux_hist_nuebar.Clone("flux_hist_nuebar"));

  // file->Add(xsec_hist_numu.Clone("xsec_hist_numu"));
  // file->Add(xsec_hist_numubar.Clone("xsec_hist_numubar"));
  // file->Add(xsec_hist_nue.Clone("xsec_hist_nue"));
  // file->Add(xsec_hist_nuebar.Clone("xsec_hist_nuebar"));

  // file->Add(Prediction_hist_numu.Clone("Prediction_hist_numu"));
  // file->Add(Prediction_hist_numubar.Clone("Prediction_hist_numubar"));
  // file->Add(Prediction_hist_nue.Clone("Prediction_hist_nue"));
  // file->Add(Prediction_hist_nuebar.Clone("Prediction_hist_nuebar"));

  // file->Write();
  // file->Close();
  // delete file;
}

void ParBinned::flip_hierarchy() {
  OscillationParameters::flip_hierarchy();
  UpdatePrediction();
}

void ParBinned::Save_prob_hist(const std::string &name) {
  // if constexpr (std::is_same_v<ParProb3ppOscillation, propgator_type>) {
  auto file = TFile::Open(name.c_str(), "RECREATE");
  file->cd();
  auto prob_hist = propagator->GetProb_Hists_3F(Ebins, costheta_bins, *this);
  auto id_2_name = std::to_array({"nue", "numu", "nutau"});
  // prob_hist: [0 neutrino, 1 antineutrino][from: 0-nue, 1-mu][to: 0-e,
  // 1-mu]
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        prob_hist[i][j][k].SetName(
            std::format("{}_{}_{}", i == 0 ? "neutrino" : "antineutrino",
                        id_2_name[j], id_2_name[k])
                .c_str());
        prob_hist[i][j][k].Write();
      }
    }
  }
  // file->Write();
  file->Close();
  delete file;
  // }
}
