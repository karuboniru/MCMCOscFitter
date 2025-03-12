// #include <__clang_cuda_intrinsics.h>
#define __MDSPAN_USE_PAREN_OPERATOR 1

#include "ParBinnedKernels.cuh"
#include "constants.h"

constexpr size_t warp_size = 32;

constexpr auto __device__ __host__ get_indexes(size_t costh_bins,
                                               size_t current_thread_id) {
  auto current_costh_analysis_bin = current_thread_id % costh_bins;
  auto current_energy_analysis_bin = current_thread_id / costh_bins;
  return std::make_pair(current_costh_analysis_bin,
                        current_energy_analysis_bin);
}

constexpr auto __device__ __host__ get_indexes_alt(size_t e_bins,
                                                   size_t current_thread_id) {
  auto current_costh_analysis_bin = current_thread_id / e_bins;
  auto current_energy_analysis_bin = current_thread_id % e_bins;
  return std::make_pair(current_costh_analysis_bin,
                        current_energy_analysis_bin);
}

void __global__ calc_event_count_and_rebin(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor) {
  auto costh_bins = ret_numu.extent(0);
  auto e_bins = ret_numu.extent(1);
  auto this_thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (this_thread_id >= (costh_bins * e_bins)) {
    return;
  }
  auto [current_costh_analysis_bin, current_energy_analysis_bin] =
      get_indexes(costh_bins, this_thread_id);

  // just tell compiler the extra information that we know the sizes
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

  ret_numu(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
  ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
  ret_nue(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
  ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;

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
      auto event_count_nue_final =
          (oscProb(0, 0, this_index_costh, this_index_E) *
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
      ret_numu(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_numu_final;
      ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_numubar_final;
      ret_nue(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_nue_final;
      ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_nuebar_final;
    }
  }
}

void __global__ calc_event_count_noosc(
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    const_vec_span xsec_nc_nu, const_vec_span xsec_nc_nubar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, span_2d_hist_t ret_nc, size_t E_rebin_factor,
    size_t costh_rebin_factor) {
  auto costh_bins = ret_numu.extent(0);
  auto e_bins = ret_numu.extent(1);
  auto this_thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (this_thread_id >= (costh_bins * e_bins)) {
    return;
  }

  // just tell compiler the extra information that we know the sizes
  if (flux_numu.extents() != flux_numubar.extents() ||
      flux_numu.extents() != flux_nue.extents() ||
      flux_numu.extents() != flux_nuebar.extents() ||
      flux_numu.extent(1) != xsec_numu.size() ||
      flux_numu.extent(1) != xsec_numubar.size() ||
      flux_numu.extent(1) != xsec_nue.size() ||
      flux_numu.extent(1) != xsec_nuebar.size() ||
      ret_numu.extents() != ret_numubar.extents() ||
      ret_numu.extents() != ret_nue.extents() ||
      ret_numu.extents() != ret_nuebar.extents() ||
      flux_numu.extent(1) != E_rebin_factor * ret_numu.extent(1) ||
      flux_numu.extent(0) != costh_rebin_factor * ret_numu.extent(0)) {
    __builtin_unreachable();
  }

  auto [current_costh_analysis_bin, current_energy_analysis_bin] =
      get_indexes(costh_bins, this_thread_id);

  ret_numu(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
  ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
  ret_nue(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
  ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;

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
          flux_numu(this_index_costh, this_index_E) * xsec_numu[this_index_E];
      auto event_count_numubar_final =
          flux_numubar(this_index_costh, this_index_E) *
          xsec_numubar[this_index_E];
      auto event_count_nue_final =
          flux_nue(this_index_costh, this_index_E) * xsec_nue[this_index_E];
      auto event_count_nuebar_final =
          flux_nuebar(this_index_costh, this_index_E) *
          xsec_nuebar[this_index_E];
      ret_numu(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_numu_final;
      ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_numubar_final;
      ret_nue(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_nue_final;
      ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin) +=
          event_count_nuebar_final;
    }
  }
}

void __global__ calc_event_count(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar) {

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
      flux_numu.extents() != ret_numu.extents()

  ) {
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
  ret_numu(this_index_costh, this_index_E) = event_count_numu_final;
  ret_numubar(this_index_costh, this_index_E) = event_count_numubar_final;
  ret_nue(this_index_costh, this_index_E) = event_count_nue_final;
  ret_nuebar(this_index_costh, this_index_E) = event_count_nuebar_final;
}

void __global__ rebinner_1(span_2d_hist_t fine_bin_numu,
                           span_2d_hist_t fine_bin_numubar,
                           span_2d_hist_t fine_bin_nue,
                           span_2d_hist_t fine_bin_nuebar,
                           size_t E_rebin_factor, size_t costh_rebin_factor) {
  auto costh_fine_bins = fine_bin_numu.extent(0);
  auto e_fine_bins = fine_bin_numu.extent(1);
  auto e_coarse_bins = e_fine_bins / E_rebin_factor;

  auto global_id = threadIdx.x + (blockDim.x * blockIdx.x);

  if (global_id < (costh_fine_bins * e_fine_bins / E_rebin_factor)) {
    auto [current_costh_fine_bin, current_energy_coarse_bin] =
        get_indexes(costh_fine_bins, global_id);
    auto current_energy_fine_bin = current_energy_coarse_bin * E_rebin_factor;
    for (size_t e_offset = 1; e_offset < E_rebin_factor; e_offset++) {
      auto current_energy_fine_bin_from = current_energy_fine_bin + e_offset;
      fine_bin_numu(current_costh_fine_bin, current_energy_fine_bin) +=
          fine_bin_numu(current_costh_fine_bin, current_energy_fine_bin_from);
      fine_bin_numubar(current_costh_fine_bin, current_energy_fine_bin) +=
          fine_bin_numubar(current_costh_fine_bin,
                           current_energy_fine_bin_from);
      fine_bin_nue(current_costh_fine_bin, current_energy_fine_bin) +=
          fine_bin_nue(current_costh_fine_bin, current_energy_fine_bin_from);
      fine_bin_nuebar(current_costh_fine_bin, current_energy_fine_bin) +=
          fine_bin_nuebar(current_costh_fine_bin, current_energy_fine_bin_from);
    }
  }
}

void __global__
rebinner_2(span_2d_hist_t fine_bin_numu, span_2d_hist_t fine_bin_numubar,
           span_2d_hist_t fine_bin_nue, span_2d_hist_t fine_bin_nuebar,
           span_2d_hist_t coarse_bin_numu, span_2d_hist_t coarse_bin_numubar,
           span_2d_hist_t coarse_bin_nue, span_2d_hist_t coarse_bin_nuebar,
           size_t E_rebin_factor, size_t costh_rebin_factor) {
  auto costh_coarse_bins = coarse_bin_numu.extent(0);
  auto e_coarse_bins = coarse_bin_numu.extent(1);

  auto global_id = threadIdx.x + (blockDim.x * blockIdx.x);
  // rebin again along cost, this time save to coarse bin span
  if (global_id < (costh_coarse_bins * e_coarse_bins)) {
    auto [current_costh_coarse_bin, current_energy_coarse_bin] =
        get_indexes(costh_coarse_bins, global_id);
    auto current_energy_fine_bin = current_energy_coarse_bin * E_rebin_factor;
    auto current_costh_fine_bin = current_costh_coarse_bin * costh_rebin_factor;
    coarse_bin_numu(current_costh_coarse_bin, current_energy_coarse_bin) = 0;
    coarse_bin_numubar(current_costh_coarse_bin, current_energy_coarse_bin) = 0;
    coarse_bin_nue(current_costh_coarse_bin, current_energy_coarse_bin) = 0;
    coarse_bin_nuebar(current_costh_coarse_bin, current_energy_coarse_bin) = 0;
    for (size_t costh_offset = 0; costh_offset < costh_rebin_factor;
         costh_offset++) {
      auto current_costh_fine_bin_from = current_costh_fine_bin + costh_offset;
      coarse_bin_numu(current_costh_coarse_bin, current_energy_coarse_bin) +=
          fine_bin_numu(current_costh_fine_bin_from, current_energy_fine_bin);
      coarse_bin_numubar(current_costh_coarse_bin, current_energy_coarse_bin) +=
          fine_bin_numubar(current_costh_fine_bin_from,
                           current_energy_fine_bin);
      coarse_bin_nue(current_costh_coarse_bin, current_energy_coarse_bin) +=
          fine_bin_nue(current_costh_fine_bin_from, current_energy_fine_bin);
      coarse_bin_nuebar(current_costh_coarse_bin, current_energy_coarse_bin) +=
          fine_bin_nuebar(current_costh_fine_bin_from, current_energy_fine_bin);
    }
  }
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