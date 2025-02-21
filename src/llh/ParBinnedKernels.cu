#define __MDSPAN_USE_PAREN_OPERATOR 1


#include "constants.h"
#include "ParBinnedKernels.cuh"

constexpr size_t warp_size = 32;

constexpr auto __device__ __host__ get_indexes(size_t costh_bins,
                                               size_t current_thread_id) {
  auto current_costh_analysis_bin = current_thread_id % costh_bins;
  auto current_energy_analysis_bin = current_thread_id / costh_bins;
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
    span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t ret_nue,
    span_2d_hist_t ret_nuebar, size_t E_rebin_factor,
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

// void __global__ calc_event_count_and_rebin_ver_calc_bin(
//     ParProb3ppOscillation::oscillaton_span_t oscProb,
//     ParProb3ppOscillation::oscillaton_span_t oscProb_anti,
//     const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
//     const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
//     const_vec_span xsec_numu, const_vec_span xsec_numubar,
//     const_vec_span xsec_nue, const_vec_span xsec_nuebar,
//     span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t
//     ret_nue, span_2d_hist_t ret_nuebar, size_t E_rebin_factor, size_t
//     costh_rebin_factor) {
// //   __shared__ oscillaton_calc_precision per_thread_event_count[4 *
// warp_size];
// //   auto shared_result_span =
// //       cuda::std::mdspan<oscillaton_calc_precision,
// //                         cuda::std::extents<size_t, 4, warp_size>>(
// //           per_thread_event_count);

//   auto costh_bins_fine = flux_numu.extent(0);
//   auto e_bins_fine = flux_numu.extent(1);
//   auto this_thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
//   if (this_thread_id >= (costh_bins_fine * e_bins_fine)) {
//     return;
//   }
//   auto [current_costh_fine_bin, current_energy_fine_bin] =
//       get_indexes(costh_bins_fine, this_thread_id);
//   auto current_costh_analysis_bin =
//       current_costh_fine_bin / costh_rebin_factor; // coarse bin index
//   auto current_energy_analysis_bin =
//       current_energy_fine_bin / E_rebin_factor; // coarse bin index

//   // just tell compiler the extra information that we know the sizes
//   if (flux_numu.extents() != flux_numubar.extents() ||
//       flux_numu.extents() != flux_nue.extents() ||
//       flux_numu.extents() != flux_nuebar.extents() ||
//       flux_numu.extent(1) != xsec_numu.size() ||
//       flux_numu.extent(1) != xsec_numubar.size() ||
//       flux_numu.extent(1) != xsec_nue.size() ||
//       flux_numu.extent(1) != xsec_nuebar.size() ||
//       oscProb.extents() != oscProb_anti.extents() ||
//       oscProb.extent(2) != flux_numu.extent(0) ||
//       oscProb.extent(3) != flux_numu.extent(1) ||
//       ret_numu.extents() != ret_numubar.extents() ||
//       ret_numu.extents() != ret_nue.extents() ||
//       ret_numu.extents() != ret_nuebar.extents() ||
//       flux_numu.extent(1) != E_rebin_factor * ret_numu.extent(1) ||
//       flux_numu.extent(0) != costh_rebin_factor * ret_numu.extent(0)) {
//     __builtin_unreachable();
//     // return;
//   }

//   ret_numu(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
//   ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
//   ret_nue(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
//   ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
//   auto this_index_costh = current_costh_fine_bin; // fine bin index
//   auto this_index_E = current_energy_fine_bin;    // fine bin index

//   auto event_count_numu_final =
//       (oscProb(0, 1, this_index_costh, this_index_E) *
//            flux_nue(this_index_costh, this_index_E) +
//        oscProb(1, 1, this_index_costh, this_index_E) *
//            flux_numu(this_index_costh, this_index_E)) *
//       xsec_numu[this_index_E];
//   auto event_count_numubar_final =
//       (oscProb_anti(0, 1, this_index_costh, this_index_E) *
//            flux_nuebar(this_index_costh, this_index_E) +
//        oscProb_anti(1, 1, this_index_costh, this_index_E) *
//            flux_numubar(this_index_costh, this_index_E)) *
//       xsec_numubar[this_index_E];
//   auto event_count_nue_final = (oscProb(0, 0, this_index_costh, this_index_E)
//   *
//                                     flux_nue(this_index_costh, this_index_E)
//                                     +
//                                 oscProb(1, 0, this_index_costh, this_index_E)
//                                 *
//                                     flux_numu(this_index_costh,
//                                     this_index_E)) *
//                                xsec_nue[this_index_E];
//   auto event_count_nuebar_final =
//       (oscProb_anti(0, 0, this_index_costh, this_index_E) *
//            flux_nuebar(this_index_costh, this_index_E) +
//        oscProb_anti(1, 0, this_index_costh, this_index_E) *
//            flux_numubar(this_index_costh, this_index_E)) *
//       xsec_nuebar[this_index_E];

// //   shared_result_span(0, threadIdx.x) = event_count_numu_final;
// //   shared_result_span(1, threadIdx.x) = event_count_numubar_final;
// //   shared_result_span(2, threadIdx.x) = event_count_nue_final;
// //   shared_result_span(3, threadIdx.x) = event_count_nuebar_final;
//     atomicAdd(&ret_numu(current_costh_analysis_bin,
//     current_energy_analysis_bin),
//               event_count_numu_final);
//     atomicAdd(
//         &ret_numubar(current_costh_analysis_bin,
//         current_energy_analysis_bin), event_count_numubar_final);
//     atomicAdd(&ret_nue(current_costh_analysis_bin,
//     current_energy_analysis_bin),
//               event_count_nue_final);
//     atomicAdd(
//         &ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin),
//         event_count_nuebar_final);
// //   __syncthreads();
// //   if (threadIdx.x == 0) {
// //     for (auto thid = 0; thid < warp_size; thid++) {
// //       auto new_thid = thid + this_thread_id;
// //       auto [current_costh_fine_bin, current_energy_fine_bin] =
// //           get_indexes(costh_bins_fine, new_thid);
// //       auto current_costh_analysis_bin =
// //           current_costh_fine_bin / costh_rebin_factor; // coarse bin index
// //       auto current_energy_analysis_bin =
// //           current_energy_fine_bin / E_rebin_factor; // coarse bin index
// //       ret_numu(current_costh_analysis_bin, current_energy_analysis_bin) +=
// //           shared_result_span(0, thid);
// //       ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin)
// +=
// //           shared_result_span(1, thid);
// //       ret_nue(current_costh_analysis_bin, current_energy_analysis_bin) +=
// //           shared_result_span(2, thid);
// //       ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin)
// +=
// //           shared_result_span(3, thid);
// //     }
// //   }
// }

// void __global__ calc_event_count_noosc_ver_calc_bin(
//     const_span_2d_hist_t flux_numu, const_span_2d_hist_t flux_numubar,
//     const_span_2d_hist_t flux_nue, const_span_2d_hist_t flux_nuebar,
//     const_vec_span xsec_numu, const_vec_span xsec_numubar,
//     const_vec_span xsec_nue, const_vec_span xsec_nuebar,
//     span_2d_hist_t ret_numu, span_2d_hist_t ret_numubar, span_2d_hist_t
//     ret_nue, span_2d_hist_t ret_nuebar, size_t E_rebin_factor, size_t
//     costh_rebin_factor) {
//   auto costh_bins_fine = flux_numu.extent(0);
//   auto e_bins_fine = flux_numu.extent(1);
//   auto this_thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
//   if (this_thread_id >= (costh_bins_fine * e_bins_fine)) {
//     return;
//   }
//   auto [current_costh_fine_bin, current_energy_fine_bin] =
//       get_indexes(costh_bins_fine, this_thread_id);
//   auto current_costh_analysis_bin =
//       current_costh_fine_bin / costh_rebin_factor; // coarse bin index
//   auto current_energy_analysis_bin =
//       current_energy_fine_bin / E_rebin_factor; // coarse bin index
//   // just tell compiler the extra information that we know the sizes
//   if (flux_numu.extents() != flux_numubar.extents() ||
//       flux_numu.extents() != flux_nue.extents() ||
//       flux_numu.extents() != flux_nuebar.extents() ||
//       flux_numu.extent(1) != xsec_numu.size() ||
//       flux_numu.extent(1) != xsec_numubar.size() ||
//       flux_numu.extent(1) != xsec_nue.size() ||
//       flux_numu.extent(1) != xsec_nuebar.size() ||
//       ret_numu.extents() != ret_numubar.extents() ||
//       ret_numu.extents() != ret_nue.extents() ||
//       ret_numu.extents() != ret_nuebar.extents() ||
//       flux_numu.extent(1) != E_rebin_factor * ret_numu.extent(1) ||
//       flux_numu.extent(0) != costh_rebin_factor * ret_numu.extent(0)) {
//     __builtin_unreachable();
//   }

//   ret_numu(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
//   ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
//   ret_nue(current_costh_analysis_bin, current_energy_analysis_bin) = 0;
//   ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin) = 0;

//   auto this_index_costh = current_costh_fine_bin; // fine bin index
//   auto this_index_E = current_energy_fine_bin;    // fine bin index

//   auto event_count_numu_final =
//       flux_numu(this_index_costh, this_index_E) * xsec_numu[this_index_E];
//   auto event_count_numubar_final =
//       flux_numubar(this_index_costh, this_index_E) *
//       xsec_numubar[this_index_E];
//   auto event_count_nue_final =
//       flux_nue(this_index_costh, this_index_E) * xsec_nue[this_index_E];
//   auto event_count_nuebar_final =
//       flux_nuebar(this_index_costh, this_index_E) *
//       xsec_nuebar[this_index_E];
//   atomicAdd(&ret_numu(current_costh_analysis_bin,
//   current_energy_analysis_bin),
//             event_count_numu_final);
//   atomicAdd(
//       &ret_numubar(current_costh_analysis_bin, current_energy_analysis_bin),
//       event_count_numubar_final);
//   atomicAdd(&ret_nue(current_costh_analysis_bin,
//   current_energy_analysis_bin),
//             event_count_nue_final);
//   atomicAdd(
//       &ret_nuebar(current_costh_analysis_bin, current_energy_analysis_bin),
//       event_count_nuebar_final);
// }
