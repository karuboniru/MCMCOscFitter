
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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include "ParProb3ppOscillation.h"
#include "Prob3ppOscillation.h"
#include "SimpleDataHist.h"
#include "constants.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"
#include <format>
#include <functional>

using propgator_type = ParProb3ppOscillation;

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
using const_vec_span = cuda::std::span<const oscillaton_calc_precision>;

void __global__ calc_event_count_and_rebin(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    span_2d_hist_t flux_numu, span_2d_hist_t flux_numubar,
    span_2d_hist_t flux_nue, span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
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
      auto this_index_costh =
          (current_costh_analysis_bin * costh_rebin_factor) +
          offset_costh; // fine bin index
      auto this_index_E = (current_energy_analysis_bin * E_rebin_factor) +
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

void __global__ calc_event_count_noosc(
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
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
      auto this_index_costh =
          (current_costh_analysis_bin * costh_rebin_factor) +
          offset_costh; // fine bin index
      auto this_index_E = (current_energy_analysis_bin * E_rebin_factor) +
                          offset_energy; // fine bin index
      auto event_count_numu_final =
          flux_numu[this_index_costh, this_index_E] * xsec_numu[this_index_E];
      auto event_count_numubar_final =
          flux_numubar[this_index_costh, this_index_E] *
          xsec_numubar[this_index_E];
      auto event_count_nue_final =
          flux_nue[this_index_costh, this_index_E] * xsec_nue[this_index_E];
      auto event_count_nuebar_final =
          flux_nuebar[this_index_costh, this_index_E] *
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

class ParBinned : public propgator_type, public ModelDataLLH {
public:
  ParBinned(std::vector<double> Ebins, std::vector<double> costheta_bins,
            double scale_ = 1., size_t E_rebin_factor = 1,
            size_t costh_rebin_factor = 1, double IH_Bias = 1.0);

  ParBinned(const ParBinned &) = default;
  ParBinned(ParBinned &&) = default;
  ParBinned &operator=(const ParBinned &) = default;
  ParBinned &operator=(ParBinned &&) = default;
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

  void flip_hierarchy() {
    OscillationParameters::flip_hierarchy();
    if constexpr (std::is_same_v<ParProb3ppOscillation, propgator_type>) {
      re_calculate();
    }
    UpdatePrediction();
  }

  void Save_prob_hist(const std::string &name) {
    // if constexpr (std::is_same_v<ParProb3ppOscillation, propgator_type>) {
    auto file = TFile::Open(name.c_str(), "RECREATE");
    file->cd();
    auto prob_hist = GetProb_Hists_3F(Ebins, costheta_bins);
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
  std::vector<double> Ebins, costheta_bins;
  // std::vector<double> Ebins_calc, costheta_bins_calc;

  // index: [cosine, energy]
  thrust::device_vector<oscillaton_calc_precision> flux_hist_numu,
      flux_hist_numubar, flux_hist_nue, flux_hist_nuebar;

  // 1D in energy
  thrust::device_vector<oscillaton_calc_precision> xsec_hist_numu,
      xsec_hist_numubar, xsec_hist_nue, xsec_hist_nuebar;

  // index: [cosine, energy]
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_numu,
      no_osc_hist_numubar, no_osc_hist_nue, no_osc_hist_nuebar;

  size_t E_rebin_factor;
  size_t costh_rebin_factor;

  size_t E_fine_bin_count, costh_fine_bin_count, E_analysis_bin_count,
      costh_analysis_bin_count;

  // index: [cosine, energy]
  thrust::device_vector<oscillaton_calc_precision> Prediction_hist_numu,
      Prediction_hist_numubar, Prediction_hist_nue, Prediction_hist_nuebar;

  double log_ih_bias;
};

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
      Prediction_hist_numu(E_analysis_bin_count * costh_analysis_bin_count),
      Prediction_hist_numubar(E_analysis_bin_count * costh_analysis_bin_count),
      Prediction_hist_nue(E_analysis_bin_count * costh_analysis_bin_count),
      Prediction_hist_nuebar(E_analysis_bin_count * costh_analysis_bin_count),
      log_ih_bias(std::log(IH_bias_))

{
  UpdatePrediction();
}

auto vec2span_1d(auto &vec) {
  return cuda::std::span<std::remove_reference_t<decltype(*vec.data().get())>>(
      vec.data().get(), vec.size());
}

void ParBinned::UpdatePrediction() {
  auto span_prob_neutrino = propgator_type::get_dev_span_neutrino();
  CUERR
  auto span_prob_antineutrino = propgator_type::get_dev_span_antineutrino();
  CUERR

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
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_numu(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_numubar(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_nue(
      E_analysis_bin_count * costh_analysis_bin_count, 0);
  thrust::device_vector<oscillaton_calc_precision> no_osc_hist_nuebar(
      E_analysis_bin_count * costh_analysis_bin_count, 0);

  dim3 block_size(E_analysis_bin_count, costh_analysis_bin_count);
  calc_event_count_noosc<<<1, block_size>>>(
      vec2span_fine(flux_hist_numu), vec2span_fine(flux_hist_numubar),
      vec2span_fine(flux_hist_nue), vec2span_fine(flux_hist_nuebar),
      vec2span_1d(xsec_hist_numu), vec2span_1d(xsec_hist_numubar),
      vec2span_1d(xsec_hist_nue), vec2span_1d(xsec_hist_nuebar),
      vec2span_analysis(no_osc_hist_numu),
      vec2span_analysis(no_osc_hist_numubar),
      vec2span_analysis(no_osc_hist_nue), vec2span_analysis(no_osc_hist_nuebar),
      E_rebin_factor, costh_rebin_factor);
  CUERR

  cudaDeviceSynchronize();
  CUERR

  data.hist_numu = vec_to_hist(no_osc_hist_numu, costh_analysis_bin_count,
                               E_analysis_bin_count);
  data.hist_numubar = vec_to_hist(no_osc_hist_numubar, costh_analysis_bin_count,
                                  E_analysis_bin_count);
  data.hist_nue = vec_to_hist(no_osc_hist_nue, costh_analysis_bin_count,
                              E_analysis_bin_count);
  data.hist_nuebar = vec_to_hist(no_osc_hist_nuebar, costh_analysis_bin_count,
                                 E_analysis_bin_count);

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

ParBinnedInterface::ParBinnedInterface(std::vector<double> Ebins,
                                       std::vector<double> costheta_bins,
                                       double scale_, size_t E_rebin_factor,
                                       size_t costh_rebin_factor,
                                       double IH_Bias)
    : pImpl(std::make_unique<ParBinned>(
          std::move(Ebins), std::move(costheta_bins), scale_, E_rebin_factor,
          costh_rebin_factor, IH_Bias)) {}

ParBinnedInterface::~ParBinnedInterface() = default;
ParBinnedInterface::ParBinnedInterface(ParBinnedInterface &&) noexcept =
    default;

ParBinnedInterface::ParBinnedInterface(const ParBinnedInterface &other)
    : pImpl(std::make_unique<ParBinned>(*other.pImpl)) {}

ParBinnedInterface &
ParBinnedInterface::operator=(ParBinnedInterface &&) noexcept = default;

ParBinnedInterface &
ParBinnedInterface::operator=(const ParBinnedInterface &other) {
  if (this != &other) {
    pImpl = std::make_unique<ParBinned>(*other.pImpl);
  }
  return *this;
}

void ParBinnedInterface::proposeStep() { pImpl->proposeStep(); }

[[nodiscard]] double ParBinnedInterface::GetLogLikelihood() const {
  return pImpl->GetLogLikelihood();
}

[[nodiscard]] double
ParBinnedInterface::GetLogLikelihoodAgainstData(const StateI &dataset) const {
  return pImpl->GetLogLikelihoodAgainstData(dataset);
}

[[nodiscard]] SimpleDataHist ParBinnedInterface::GenerateData() const {
  return pImpl->GenerateData();
}

[[nodiscard]] SimpleDataHist ParBinnedInterface::GenerateData_NoOsc() const {
  return pImpl->GenerateData_NoOsc();
}

void ParBinnedInterface::Print() const { pImpl->Print(); }

void ParBinnedInterface::flip_hierarchy() { pImpl->flip_hierarchy(); }

void ParBinnedInterface::Save_prob_hist(const std::string &name) {
  pImpl->Save_prob_hist(name);
}

void ParBinnedInterface::SaveAs(const char *filename) const {
  pImpl->SaveAs(filename);
}

void ParBinnedInterface::UpdatePrediction() { pImpl->UpdatePrediction(); }

double ParBinnedInterface::GetDM32sq() const { return pImpl->GetDM32sq(); }
double ParBinnedInterface::GetDM21sq() const { return pImpl->GetDM21sq(); }
double ParBinnedInterface::GetT23() const { return pImpl->GetT23(); }
double ParBinnedInterface::GetT13() const { return pImpl->GetT13(); }
double ParBinnedInterface::GetT12() const { return pImpl->GetT12(); }
double ParBinnedInterface::GetDeltaCP() const { return pImpl->GetDeltaCP(); }

void ParBinnedInterface::set_param(const param &p) { pImpl->set_param(p); }

void ParBinnedInterface::set_toggle(const pull_toggle &t) {
  pImpl->set_toggle(t);
}
const pull_toggle &ParBinnedInterface::get_toggle() const {
  return pImpl->get_toggle();
}
