#include <constants.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include "ParProb3ppOscillation.h"

using vec_span = cuda::std::span<oscillaton_calc_precision>;
using const_vec_span = cuda::std::span<const oscillaton_calc_precision>;
using span_2d_hist_t =
    cuda::std::mdspan<oscillaton_calc_precision,
                      cuda::std::extents<size_t, cuda::std::dynamic_extent,
                                         cuda::std::dynamic_extent>>;
using const_span_2d_hist_t =
    cuda::std::mdspan<const oscillaton_calc_precision,
                      cuda::std::extents<size_t, cuda::std::dynamic_extent,
                                         cuda::std::dynamic_extent>>;

void __global__ calc_event_count(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar);

void __global__ calc_event_count_and_rebin(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor);

void __global__ calc_event_count_noosc(
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor);

void __global__ rebinner_1(span_2d_hist_t fine_bin_numu,
                           span_2d_hist_t fine_bin_numubar,
                           span_2d_hist_t fine_bin_nue,
                           span_2d_hist_t fine_bin_nuebar,
                           size_t E_rebin_factor, size_t costh_rebin_factor);

void __global__
rebinner_2(span_2d_hist_t fine_bin_numu, span_2d_hist_t fine_bin_numubar,
           span_2d_hist_t fine_bin_nue, span_2d_hist_t fine_bin_nuebar,
           span_2d_hist_t coarse_bin_numu, span_2d_hist_t coarse_bin_numubar,
           span_2d_hist_t coarse_bin_nue, span_2d_hist_t coarse_bin_nuebar,
           size_t E_rebin_factor, size_t costh_rebin_factor);

void __global__ calc_event_count_atomic_add(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor);

void __global__ calc_event_count_atomic_add(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    const oscillaton_calc_precision *oscProb_anti,
    const oscillaton_calc_precision *flux_numu,
    const oscillaton_calc_precision *flux_numubar,
    const oscillaton_calc_precision *flux_nue,
    const oscillaton_calc_precision *flux_nuebar,
    const oscillaton_calc_precision *xsec_numu,
    const oscillaton_calc_precision *xsec_numubar,
    const oscillaton_calc_precision *xsec_nue,
    const oscillaton_calc_precision *xsec_nuebar,
    oscillaton_calc_precision *ret_numu, oscillaton_calc_precision *ret_numubar,
    oscillaton_calc_precision *ret_nue, oscillaton_calc_precision *ret_nuebar,
    size_t E_rebin_factor, size_t costh_rebin_factor);

void __global__ calc_event_count_noosc_atomic(
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor);

void __global__ calc_event_count_reduction(
    ParProb3ppOscillation::oscillaton_span_t oscProb,
    ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
    const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
    const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
    const_vec_span xsec_numu, const_vec_span xsec_numubar,
    const_vec_span xsec_nue, const_vec_span xsec_nuebar,
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
    size_t costh_rebin_factor);