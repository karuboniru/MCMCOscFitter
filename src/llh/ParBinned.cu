
#include "OscillationParameters.h"
#include "ParBinned.h"
#include "ParProb3ppOscillation.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"

#include <SimpleDataHist.h>
#include <TF3.h>
#include <TH2.h>
#include <type_traits>
#include <utility>

#include <cuda/std/mdspan>
#include <cuda/std/span>

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
// just to avoid any potential conflict
namespace {
template <typename T>
concept is_hist = std::is_base_of_v<TH1, T>;

template <typename T>
concept is_hist2D = std::is_base_of_v<TH2, T>;

template <is_hist T> T operator*(const T &lhs, const T &rhs) {
  T ret = lhs;
  ret.Multiply(&rhs);
  return ret;
}

TH2D operator*(const TH2D &lhs, const TH1D &rhs) {
  TH2D ret = lhs;

  assert(lhs.GetNbinsX() == rhs.GetNbinsX());

  for (int x_index = 1; x_index <= lhs.GetNbinsX(); x_index++) {
    for (int y_index = 1; y_index <= lhs.GetNbinsY(); y_index++) {
      ret.SetBinContent(x_index, y_index,
                        ret.GetBinContent(x_index, y_index) *
                            rhs.GetBinContent(x_index));
    }
  }
  return ret;
}

template <is_hist T> T operator+(const T &lhs, const T &rhs) {
  T ret = lhs;
  ret.Add(&rhs);
  return ret;
}

template <is_hist T> T operator*(const T &lhs, double rhs) {
  T ret = lhs;
  ret.Scale(rhs);
  return ret;
}

template <is_hist T> T operator*(double lhs, const T &&rhs) {
  return rhs * lhs;
}

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
      ret_span[y - 1, x - 1] = hist.GetBinContent(x, y);
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
      ret.SetBinContent(x, y, ret_span[y - 1, x - 1]);
    }
  }
  return ret;
}

using vec_span = cuda::std::span<oscillaton_calc_precision>;

void __global__ calc_event_count_and_rebin(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    span_2d_hist_t flux_numu, span_2d_hist_t flux_numubar,
    span_2d_hist_t flux_nue, span_2d_hist_t flux_nuebar, vec_span xsec_numu,
    vec_span xsec_numubar, vec_span xsec_nue, vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor) {

  auto current_energy_analysis_bin = threadIdx.x;
  auto current_costh_analysis_bin = threadIdx.y;
  ret_numu[current_costh_analysis_bin, current_energy_analysis_bin] =
      ret_numubar[current_costh_analysis_bin, current_energy_analysis_bin] =
          ret_nue[current_costh_analysis_bin, current_energy_analysis_bin] =
              ret_nuebar[current_costh_analysis_bin,
                         current_energy_analysis_bin] = 0;

  for (size_t offset_costh = 0; offset_costh < costh_rebin_factor;
       offset_costh++) {
    for (size_t offset_energy = 0; offset_energy < E_rebin_factor;
         offset_energy++) {
      auto this_index_costh = current_costh_analysis_bin * costh_rebin_factor +
                              offset_costh; // fine bin index
      auto this_index_E = current_energy_analysis_bin * E_rebin_factor +
                          offset_energy; // fine bin index
      auto event_count_numu_final =
          (oscProb[0, 1, this_index_costh, this_index_E] *
               flux_nue[this_index_costh, this_index_E] +
           oscProb[1, 1, this_index_costh, this_index_E] *
               flux_numu[this_index_costh, this_index_E]) *
          xsec_numu[this_index_E];
      auto event_count_numubar_final =
          (oscProb_anti[0, 1, this_index_costh, this_index_E] *
               flux_nuebar[this_index_costh, this_index_E] +
           oscProb_anti[1, 1, this_index_costh, this_index_E] *
               flux_numubar[this_index_costh, this_index_E]) *
          xsec_numubar[this_index_E];
      auto event_count_nue_final =
          (oscProb[0, 0, this_index_costh, this_index_E] *
               flux_nue[this_index_costh, this_index_E] +
           oscProb[1, 0, this_index_costh, this_index_E] *
               flux_numu[this_index_costh, this_index_E]) *
          xsec_nue[this_index_E];
      auto event_count_nuebar_final =
          (oscProb_anti[0, 0, this_index_costh, this_index_E] *
               flux_nuebar[this_index_costh, this_index_E] +
           oscProb_anti[1, 0, this_index_costh, this_index_E] *
               flux_numubar[this_index_costh, this_index_E]) *
          xsec_nuebar[this_index_E];
      ret_numu[current_costh_analysis_bin, current_energy_analysis_bin] +=
          event_count_numu_final;
      ret_numubar[current_costh_analysis_bin, current_energy_analysis_bin] +=
          event_count_numubar_final;
      ret_nue[current_costh_analysis_bin, current_energy_analysis_bin] +=
          event_count_nue_final;
      ret_nuebar[current_costh_analysis_bin, current_energy_analysis_bin] +=
          event_count_nuebar_final;
    }
  }
}
} // namespace

ParBinned::ParBinned(std::vector<double> Ebins_,
                     std::vector<double> costheta_bins_, double scale_,
                     size_t E_rebin_factor_, size_t costh_rebin_factor_,
                     double IH_bias_)
    : propgator_type{to_center<oscillaton_calc_precision>(Ebins_),
                     to_center<oscillaton_calc_precision>(costheta_bins_)},
      ModelDataLLH(), Ebins(std::move(Ebins_)),
      costheta_bins(std::move(costheta_bins_)),
      flux_hist_numu(TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, 14) * scale_)),
      flux_hist_numubar(TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, -14) * scale_)),
      flux_hist_nue(TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, 12) * scale_)),
      flux_hist_nuebar(TH2D_to_hist(
          flux_input.GetFlux_Hist(Ebins, costheta_bins, -12) * scale_)),
      xsec_hist_numu(TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, 14, {{1000060120, 1.0}, {2212, H_to_C}}))),
      xsec_hist_numubar(TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, -14, {{1000060120, 1.0}, {2212, H_to_C}}))),
      xsec_hist_nue(TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}}))),
      xsec_hist_nuebar(TH1_to_hist(xsec_input.GetXsecHistMixture(
          Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}}))),
      E_rebin_factor(E_rebin_factor_), costh_rebin_factor(costh_rebin_factor_),
      E_fine_bin_count(Ebins.size() - 1),
      costh_fine_bin_count(costheta_bins.size() - 1),
      E_analysis_bin_count(E_fine_bin_count / E_rebin_factor),
      costh_analysis_bin_count(costh_fine_bin_count / costh_rebin_factor),
      Prediction_hist_numu(E_analysis_bin_count * costh_fine_bin_count, 0),
      Prediction_hist_numubar(E_analysis_bin_count * costh_fine_bin_count, 0),
      Prediction_hist_nue(E_analysis_bin_count * costh_fine_bin_count, 0),
      Prediction_hist_nuebar(E_analysis_bin_count * costh_fine_bin_count, 0),
      log_ih_bias(std::log(IH_bias_))

{
  UpdatePrediction();
}

auto vec2span_1d(auto &vec) {
  return cuda::std::span<oscillaton_calc_precision>(
      thrust::raw_pointer_cast(vec.data()), vec.size());
}

void ParBinned::UpdatePrediction() {
  auto span_prob_neutrino = propgator_type::get_dev_span_neutrino();
  CUERR
  auto span_prob_antineutrino = propgator_type::get_dev_span_antineutrino();
  CUERR

  // thrust::

  // dim3 calc_grid(costh_fine_bin_count, E_fine_bin_count);
  // auto thread_per_block = E_rebin_factor * costh_rebin_factor;
  dim3 block_size(E_analysis_bin_count, costh_analysis_bin_count);

  calc_event_count_and_rebin<<<1, block_size>>>(
      span_prob_neutrino, span_prob_antineutrino, vec2span_fine(flux_hist_numu),
      vec2span_fine(flux_hist_numubar), vec2span_fine(flux_hist_nue),
      vec2span_fine(flux_hist_nuebar), vec2span_1d(xsec_hist_numu),
      vec2span_1d(xsec_hist_numubar), vec2span_1d(xsec_hist_nue),
      vec2span_1d(xsec_hist_nuebar), vec2span_analysis(Prediction_hist_numu),
      vec2span_analysis(Prediction_hist_numubar),
      vec2span_analysis(Prediction_hist_nue),
      vec2span_analysis(Prediction_hist_nuebar), E_rebin_factor,
      costh_rebin_factor);
  CUERR

  cudaDeviceSynchronize();
  CUERR
}

void ParBinned::proposeStep() {
  propgator_type::proposeStep();
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
      TH2D_chi2(data_hist_numu,
                vec_to_hist(Prediction_hist_numu, costh_analysis_bin_count,
                            E_analysis_bin_count));
  auto chi2_numubar =
      TH2D_chi2(data_hist_numubar,
                vec_to_hist(Prediction_hist_numubar, costh_analysis_bin_count,
                            E_analysis_bin_count));
  auto chi2_nue = TH2D_chi2(data_hist_nue, vec_to_hist(Prediction_hist_nue,
                                                       costh_analysis_bin_count,
                                                       E_analysis_bin_count));
  auto chi2_nuebar =
      TH2D_chi2(data_hist_nuebar,
                vec_to_hist(Prediction_hist_nuebar, costh_analysis_bin_count,
                            E_analysis_bin_count));

  auto llh = -0.5 * (chi2_numu + chi2_numubar + chi2_nue + chi2_nuebar);
  return llh;
}

SimpleDataHist ParBinned::GenerateData() const {
  SimpleDataHist data;
  data.hist_numu = vec_to_hist(Prediction_hist_numu, costh_analysis_bin_count,
                               E_analysis_bin_count);
  data.hist_numubar = vec_to_hist(
      Prediction_hist_numubar, costh_analysis_bin_count, E_analysis_bin_count);
  data.hist_nue = vec_to_hist(Prediction_hist_nue, costh_analysis_bin_count,
                              E_analysis_bin_count);
  data.hist_nuebar = vec_to_hist(
      Prediction_hist_nuebar, costh_analysis_bin_count, E_analysis_bin_count);
  return data;
}

SimpleDataHist ParBinned::GenerateData_NoOsc() const {
  SimpleDataHist data;
  // data.hist_numu = flux_hist_numu * xsec_hist_numu;
  // data.hist_numubar = flux_hist_numubar * xsec_hist_numubar;
  // data.hist_nue = flux_hist_nue * xsec_hist_nue;
  // data.hist_nuebar = flux_hist_nuebar * xsec_hist_nuebar;
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