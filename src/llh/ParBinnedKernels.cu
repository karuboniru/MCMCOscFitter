#define __MDSPAN_USE_PAREN_OPERATOR 1

#include "ParBinnedKernels.cuh"
#include "constants.h"

constexpr auto __device__ __host__ get_indexes(size_t costh_bins,
                                                size_t current_thread_id) {
  auto current_costh_analysis_bin = current_thread_id % costh_bins;
  auto current_energy_analysis_bin = current_thread_id / costh_bins;
  return std::make_pair(current_costh_analysis_bin,
                        current_energy_analysis_bin);
}

void __global__ calc_event_count_atomic_add(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor) {

  if (flux_numu.extents() != flux_numubar.extents() ||
      flux_numu.extents() != flux_nue.extents() ||
      flux_numu.extents() != flux_nuebar.extents() ||
      flux_numu.extent(1) != xsec_numu.size() ||
      flux_numu.extent(1) != xsec_numubar.size() ||
      flux_numu.extent(1) != xsec_nue.size() ||
      flux_numu.extent(1) != xsec_nuebar.size() ||
      oscProb.extents() != oscProb_anti.extents() ||
      oscProb.extent(2) != flux_numu.extent(0) ||
      oscProb.extent(3) != flux_numu.extent(1) ||
      ret_numu.extents() != ret_numubar.extents() ||
      ret_numu.extents() != ret_nue.extents() ||
      ret_numu.extents() != ret_nuebar.extents() ||
      flux_numu.extent(1) != E_rebin_factor * ret_numu.extent(1) ||
      flux_numu.extent(0) != costh_rebin_factor * ret_numu.extent(0)) {
    __builtin_unreachable();
    // return;
  }
  auto global_thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
  auto costh_bins = flux_numu.extent(0);
  auto e_bins = flux_numu.extent(1);
  if (global_thread_id >= (costh_bins * e_bins)) {
    return;
  }
  auto [this_index_costh, this_index_E] =
      get_indexes(costh_bins, global_thread_id);
  auto event_count_numu_final =
      (oscProb(0, 1, this_index_costh, this_index_E) *
           flux_nue(this_index_costh, this_index_E) +
       oscProb(1, 1, this_index_costh, this_index_E) *
           flux_numu(this_index_costh, this_index_E)) *
      xsec_numu[this_index_E];
  auto event_count_numubar_final =
      (oscProb_anti(0, 1, this_index_costh, this_index_E) *
           flux_nuebar(this_index_costh, this_index_E) +
       oscProb_anti(1, 1, this_index_costh, this_index_E) *
           flux_numubar(this_index_costh, this_index_E)) *
      xsec_numubar[this_index_E];
  auto event_count_nue_final = (oscProb(0, 0, this_index_costh, this_index_E) *
                                    flux_nue(this_index_costh, this_index_E) +
                                oscProb(1, 0, this_index_costh, this_index_E) *
                                    flux_numu(this_index_costh, this_index_E)) *
                               xsec_nue[this_index_E];
  auto event_count_nuebar_final =
      (oscProb_anti(0, 0, this_index_costh, this_index_E) *
           flux_nuebar(this_index_costh, this_index_E) +
       oscProb_anti(1, 0, this_index_costh, this_index_E) *
           flux_numubar(this_index_costh, this_index_E)) *
      xsec_nuebar[this_index_E];
  auto target_index_costh = this_index_costh / costh_rebin_factor;
  auto target_index_E = this_index_E / E_rebin_factor;
  atomicAdd(&ret_numu(target_index_costh, target_index_E),
            event_count_numu_final);
  atomicAdd(&ret_numubar(target_index_costh, target_index_E),
            event_count_numubar_final);
  atomicAdd(&ret_nue(target_index_costh, target_index_E),
            event_count_nue_final);
  atomicAdd(&ret_nuebar(target_index_costh, target_index_E),
            event_count_nuebar_final);
}

void __global__ calc_event_count_noosc_atomic_add(
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    const_vec_span xsec_nc_nu, const_vec_span xsec_nc_nubar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, span_2d_hist_t ret_nc, size_t E_rebin_factor,
    size_t costh_rebin_factor) {
  auto global_thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
  auto costh_bins = flux_numu.extent(0);
  auto e_bins = flux_numu.extent(1);
  if (global_thread_id >= (costh_bins * e_bins)) {
    return;
  }
  auto [this_index_costh, this_index_E] =
      get_indexes(costh_bins, global_thread_id);
  auto event_count_numu_final =
      flux_numu(this_index_costh, this_index_E) * xsec_numu[this_index_E];
  auto event_count_numubar_final =
      flux_numubar(this_index_costh, this_index_E) * xsec_numubar[this_index_E];
  auto event_count_nue_final =
      flux_nue(this_index_costh, this_index_E) * xsec_nue[this_index_E];
  auto event_count_nuebar_final =
      flux_nuebar(this_index_costh, this_index_E) * xsec_nuebar[this_index_E];
  auto event_nc_final = ((flux_numu(this_index_costh, this_index_E) +
                          flux_nue(this_index_costh, this_index_E)) *
                         xsec_nc_nu[this_index_E]) +
                        ((flux_numubar(this_index_costh, this_index_E) +
                          flux_nuebar(this_index_costh, this_index_E)) *
                         xsec_nc_nubar[this_index_E]);
  auto target_index_costh = this_index_costh / costh_rebin_factor;
  auto target_index_E = this_index_E / E_rebin_factor;
  atomicAdd(&ret_numu(target_index_costh, target_index_E),
            event_count_numu_final);
  atomicAdd(&ret_numubar(target_index_costh, target_index_E),
            event_count_numubar_final);
  atomicAdd(&ret_nue(target_index_costh, target_index_E),
            event_count_nue_final);
  atomicAdd(&ret_nuebar(target_index_costh, target_index_E),
            event_count_nuebar_final);
  atomicAdd(&ret_nc(target_index_costh, target_index_E), event_nc_final);
}