#include "ParProb3ppOscillation.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"

#include <SimpleDataHist.h>
#include <TF3.h>
#include <TH2.h>
#include <cmath>
#include <mutex>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include "ParProb3ppOscillation.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataHist.h"
#include "constants.h"
#include "hondaflux2d.h"
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
      throw std::runtime_error(                                                \
          std::format("CUDA error: {} : {}, line {}",                          \
                      cudaGetErrorString(err), __FILE__, __LINE__));           \
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

// Flux and cross-section input data, resident on GPU for the
// lifetime of the process.  Initialised once; subsequent calls to
// initialize() with different parameters throw.
class global_device_input_instance {
public:
  static global_device_input_instance &get_instance() {
    static global_device_input_instance instance{};
    return instance;
  }

  void initialize(const std::vector<double> &Ebins,
                  const std::vector<double> &costheta_bins, double scale_) {
    std::call_once(init_flag_, [&] {
      stored_Ebins_ = Ebins;
      stored_costheta_bins_ = costheta_bins;
      E_fine_bin_count = Ebins.size() - 1;
      costh_fine_bin_count = costheta_bins.size() - 1;

      flux_hist_numu = TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, 14) * scale_);
      flux_hist_numubar = TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, -14) * scale_);
      flux_hist_nue = TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, 12) * scale_);
      flux_hist_nuebar = TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, -12) * scale_);

      xsec_hist_numu = TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, 14, {{1000060120, 1.0}, {2212, H_to_C}}));
      xsec_hist_numubar = TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, -14, {{1000060120, 1.0}, {2212, H_to_C}}));
      xsec_hist_nue = TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}}));
      xsec_hist_nuebar = TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}}));
      xsec_hist_nc_nu = TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}}, false));
      xsec_hist_nc_nu_bar = TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}}, false));
    });

    if (Ebins != stored_Ebins_ || costheta_bins != stored_costheta_bins_) {
      throw std::runtime_error(
          "global_device_input_instance already initialized with different "
          "parameters");
    }
  }

  [[nodiscard]] bool is_initialized() const {
    return E_fine_bin_count > 0;
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
  [[nodiscard]] auto get_xsec_nc_nu() const {
    return vec2span_1d(xsec_hist_nc_nu);
  }
  [[nodiscard]] auto get_xsec_nc_nu_bar() const {
    return vec2span_1d(xsec_hist_nc_nu_bar);
  }

  ~global_device_input_instance() = default;
  global_device_input_instance(const global_device_input_instance &) = delete;
  global_device_input_instance(global_device_input_instance &&) = delete;
  global_device_input_instance &
  operator=(const global_device_input_instance &) = delete;
  global_device_input_instance &
  operator=(global_device_input_instance &&) = delete;

private:
  global_device_input_instance() = default;

  std::once_flag init_flag_;
  std::vector<double> stored_Ebins_, stored_costheta_bins_;

  // index: [cosine, energy]
  thrust::device_vector<oscillaton_calc_precision> flux_hist_numu,
      flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;

  // 1D in energy
  thrust::device_vector<oscillaton_calc_precision> xsec_hist_numu,
      xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  thrust::device_vector<oscillaton_calc_precision> xsec_hist_nc_nu,
      xsec_hist_nc_nu_bar;

  size_t E_fine_bin_count{}, costh_fine_bin_count{};
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
    : propagator{std::make_shared<ParProb3ppOscillation>(
          to_center<oscillaton_calc_precision>(Ebins_),
          to_center<oscillaton_calc_precision>(costheta_bins_))},
      Ebins(std::move(Ebins_)), costheta_bins(std::move(costheta_bins_)),
      Ebins_analysis(stride(Ebins, E_rebin_factor_)),
      costheta_analysis(stride(costheta_bins, costh_rebin_factor_)),
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
  global_device_input_instance::get_instance().initialize(Ebins, costheta_bins,
                                                          scale_);
  UpdatePrediction();
}

ParBinned::ParBinned(const ParBinned &other)
    : OscillationParameters(other),
      propagator(other.propagator),
      Ebins(other.Ebins), costheta_bins(other.costheta_bins),
      Ebins_analysis(other.Ebins_analysis), costheta_analysis(other.costheta_analysis),
      E_rebin_factor(other.E_rebin_factor), costh_rebin_factor(other.costh_rebin_factor),
      E_fine_bin_count(other.E_fine_bin_count), costh_fine_bin_count(other.costh_fine_bin_count),
      E_analysis_bin_count(other.E_analysis_bin_count), costh_analysis_bin_count(other.costh_analysis_bin_count),
      Prediction_hist_numu(other.Prediction_hist_numu.size()),
      Prediction_hist_numubar(other.Prediction_hist_numubar.size()),
      Prediction_hist_nue(other.Prediction_hist_nue.size()),
      Prediction_hist_nuebar(other.Prediction_hist_nuebar.size()),
      log_ih_bias(other.log_ih_bias) {}

ParBinned &ParBinned::operator=(const ParBinned &other) {
  if (this != &other) {
    OscillationParameters::operator=(other);
    propagator = other.propagator;
    Ebins = other.Ebins; costheta_bins = other.costheta_bins;
    Ebins_analysis = other.Ebins_analysis; costheta_analysis = other.costheta_analysis;
    E_rebin_factor = other.E_rebin_factor; costh_rebin_factor = other.costh_rebin_factor;
    E_fine_bin_count = other.E_fine_bin_count; costh_fine_bin_count = other.costh_fine_bin_count;
    E_analysis_bin_count = other.E_analysis_bin_count; costh_analysis_bin_count = other.costh_analysis_bin_count;
    Prediction_hist_numu.resize(other.Prediction_hist_numu.size());
    Prediction_hist_numubar.resize(other.Prediction_hist_numubar.size());
    Prediction_hist_nue.resize(other.Prediction_hist_nue.size());
    Prediction_hist_nuebar.resize(other.Prediction_hist_nuebar.size());
    log_ih_bias = other.log_ih_bias;
  }
  return *this;
}

void ParBinned::UpdatePrediction() {
  propagator->re_calculate_device(*this);

  // Neutrino and antineutrino oscillation kernels run on separate CUDA
  // streams (each calculator owns its own).  The rebinning kernel below
  // reads from both d_results_ buffers, so both must be complete first.
  cudaDeviceSynchronize();

  auto span_prob_neutrino = propagator->get_dev_span_neutrino();
  CUERR
  auto span_prob_antineutrino = propagator->get_dev_span_antineutrino();
  CUERR

  auto &flux_xsec_device_input = global_device_input_instance::get_instance();

  cudaMemsetAsync(Prediction_hist_numu.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_numu.size(),
                  getStream());
  cudaMemsetAsync(Prediction_hist_numubar.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_numubar.size(),
                  getStream());
  cudaMemsetAsync(Prediction_hist_nue.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_nue.size(),
                  getStream());
  cudaMemsetAsync(Prediction_hist_nuebar.data().get(), 0,
                  sizeof(oscillaton_calc_precision) *
                      Prediction_hist_nuebar.size(),
                  getStream());

  calc_event_count_atomic_add<<<
      cuda::ceil_div(E_fine_bin_count * costh_fine_bin_count, warp_size),
      warp_size, 0, getStream()>>>(span_prob_neutrino, span_prob_antineutrino,
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
  CUERR

  // Ensure prediction buffers are ready before any host-side read.
  cudaDeviceSynchronize();
}

void ParBinned::proposeStep() {
  OscillationParameters::proposeStep();
  UpdatePrediction();
}

void ParBinned::proposeStep(std::mt19937 &rng) {
  OscillationParameters::proposeStep(rng);
  UpdatePrediction();
}

namespace {
// Copy device_vector prediction to host PodHist2D<double> for chi2.
PodHist2D<double>
dev_pred_to_pod(const thrust::device_vector<oscillaton_calc_precision> &dev,
                size_t n_costh, size_t n_e) {
  PodHist2D<double> pod(n_costh, n_e);
  // thrust::copy with implicit float→double conversion (element-wise safe).
  thrust::copy(dev.begin(), dev.end(), pod.data.begin());
  return pod;
}
} // namespace
double ParBinned::GetLogLikelihoodAgainstData(const SimpleDataHist &dataset) const {
  // Convert prediction from GPU to host PodHist2D (avoids TH2D construction).
  auto pred_numu    = dev_pred_to_pod(Prediction_hist_numu,    costh_analysis_bin_count, E_analysis_bin_count);
  auto pred_numubar = dev_pred_to_pod(Prediction_hist_numubar, costh_analysis_bin_count, E_analysis_bin_count);
  auto pred_nue     = dev_pred_to_pod(Prediction_hist_nue,     costh_analysis_bin_count, E_analysis_bin_count);
  auto pred_nuebar  = dev_pred_to_pod(Prediction_hist_nuebar,  costh_analysis_bin_count, E_analysis_bin_count);

  auto chi2_numu    = pod_chi2(dataset.data_numu,    pred_numu);
  auto chi2_numubar = pod_chi2(dataset.data_numubar, pred_numubar);
  auto chi2_nue     = pod_chi2(dataset.data_nue,     pred_nue);
  auto chi2_nuebar  = pod_chi2(dataset.data_nuebar,  pred_nuebar);

  return -0.5 * (chi2_numu + chi2_numubar + chi2_nue + chi2_nuebar);
}

SimpleDataHist ParBinned::GenerateData() const {
  SimpleDataHist data;
  data.data_numu    = dev_pred_to_pod(Prediction_hist_numu,    costh_analysis_bin_count, E_analysis_bin_count);
  data.data_numubar = dev_pred_to_pod(Prediction_hist_numubar, costh_analysis_bin_count, E_analysis_bin_count);
  data.data_nue     = dev_pred_to_pod(Prediction_hist_nue,     costh_analysis_bin_count, E_analysis_bin_count);
  data.data_nuebar  = dev_pred_to_pod(Prediction_hist_nuebar,  costh_analysis_bin_count, E_analysis_bin_count);
  data.Ebins         = Ebins_analysis;
  data.costheta_bins = costheta_analysis;
  return data;
}

SimpleDataHist ParBinned::GenerateData_NoOsc() const {
  SimpleDataHist data;
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_numu(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_numubar(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_nue(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_nuebar(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> nc(
      E_analysis_bin_count * costh_analysis_bin_count, 0);

  auto &flux_xsec_device_input = global_device_input_instance::get_instance();
  auto warp_count =
      cuda::ceil_div(costh_fine_bin_count * E_fine_bin_count, warp_size);
  calc_event_count_noosc_atomic_add<<<warp_count, warp_size>>>(
      flux_xsec_device_input.get_flux_numu(),
      flux_xsec_device_input.get_flux_numubar(),
      flux_xsec_device_input.get_flux_nue(),
      flux_xsec_device_input.get_flux_nuebar(),
      flux_xsec_device_input.get_xsec_numu(),
      flux_xsec_device_input.get_xsec_numubar(),
      flux_xsec_device_input.get_xsec_nue(),
      flux_xsec_device_input.get_xsec_nuebar(),
      flux_xsec_device_input.get_xsec_nc_nu(),
      flux_xsec_device_input.get_xsec_nc_nu_bar(),
      vec2span_analysis(no_osc_hist_numu),
      vec2span_analysis(no_osc_hist_numubar),
      vec2span_analysis(no_osc_hist_nue), vec2span_analysis(no_osc_hist_nuebar),
      vec2span_analysis(nc), E_rebin_factor, costh_rebin_factor);
  // cudaDeviceSynchronize();
  CUERR

  data.data_numu    = dev_pred_to_pod(no_osc_hist_numu,    costh_analysis_bin_count, E_analysis_bin_count);
  data.data_numubar = dev_pred_to_pod(no_osc_hist_numubar, costh_analysis_bin_count, E_analysis_bin_count);
  data.data_nue     = dev_pred_to_pod(no_osc_hist_nue,     costh_analysis_bin_count, E_analysis_bin_count);
  data.data_nuebar  = dev_pred_to_pod(no_osc_hist_nuebar,  costh_analysis_bin_count, E_analysis_bin_count);
  data.data_nc      = dev_pred_to_pod(nc, costh_analysis_bin_count, E_analysis_bin_count);
  data.Ebins         = Ebins_analysis;
  data.costheta_bins = costheta_analysis;
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
  // Not implemented for GPU path yet.
  (void)filename;
}

void ParBinned::flip_hierarchy() {
  OscillationParameters::flip_hierarchy();
  UpdatePrediction();
}

void ParBinned::Save_prob_hist(const std::string &name) {
  auto file = TFile::Open(name.c_str(), "RECREATE");
  file->cd();
  auto pod = propagator->GetProb_Hists_3F_POD(Ebins, costheta_bins, *this);
  auto id_2_name = std::to_array({"nue", "numu", "nutau"});
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        auto h = pod[i][j][k].to_th2d(Ebins, costheta_bins);
        h.SetName(std::format("{}_{}_{}", i == 0 ? "neutrino" : "antineutrino",
                              id_2_name[j], id_2_name[k])
                      .c_str());
        h.Write();
      }
    }
  }
  file->Close();
  delete file;
}
