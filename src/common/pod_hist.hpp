#pragma once

#include "constants.h"
#include <vector>
#include <TH1.h>
#include <TH2.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

/// POD 2D histogram — flat contiguous storage, layout [costh][E].
/// This matches the existing GPU-path convention (thrust::device_vector,
/// cuda::mdspan with costh as outer dimension).
///
/// Direct access:  data[costh_idx * n_e + energy_idx]
/// For mdspan views (host: std::mdspan, device: cuda::std::mdspan),
/// construct from raw pointer at the call site.
template <typename T>
struct PodHist2D {
  std::vector<T> data;
  size_t n_costh{};
  size_t n_e{};

  PodHist2D() = default;

  PodHist2D(size_t costh_bins, size_t e_bins)
      : data(costh_bins * e_bins), n_costh(costh_bins), n_e(e_bins) {}

  // ── direct element access ─────────────────────────────────────────────
  [[nodiscard]] T &      operator()(size_t costh_idx, size_t e_idx)       { return data[costh_idx * n_e + e_idx]; }
  [[nodiscard]] const T &operator()(size_t costh_idx, size_t e_idx) const { return data[costh_idx * n_e + e_idx]; }

  [[nodiscard]] T *      data_ptr()       { return data.data(); }
  [[nodiscard]] const T *data_ptr() const { return data.data(); }

  [[nodiscard]] size_t size() const { return data.size(); }
  [[nodiscard]] bool   empty() const { return data.empty(); }

  // ── ROOT TH2D conversion (I/O boundary only) ──────────────────────────
  // ROOT TH2D uses (x=E, y=costh) axis ordering.
  // PodHist2D uses (costh, E) to match existing GPU-path convention.

  /// Create from ROOT TH2D by copying bin contents.
  static PodHist2D from_th2d(const TH2D &h) {
    const int nx = h.GetNbinsX(); // E
    const int ny = h.GetNbinsY(); // costh
    PodHist2D ret(static_cast<size_t>(ny), static_cast<size_t>(nx));
    for (int x = 1; x <= nx; ++x)
      for (int y = 1; y <= ny; ++y)
        ret.data[static_cast<size_t>(y - 1) * ret.n_e +
                 static_cast<size_t>(x - 1)] =
            static_cast<T>(h.GetBinContent(x, y));
    return ret;
  }

  /// Convert to ROOT TH2D with user-supplied bin edges.
  [[nodiscard]] TH2D to_th2d(const std::vector<double> &Ebins,
                             const std::vector<double> &costhbins,
                             const char *name = "") const {
    assert(Ebins.size() >= 2);
    assert(costhbins.size() >= 2);
    const int n_ex = static_cast<int>(Ebins.size()) - 1;
    const int n_cx = static_cast<int>(costhbins.size()) - 1;
    TH2D h(name, "", n_ex, Ebins.data(), n_cx, costhbins.data());
    for (int x = 1; x <= n_ex; ++x)        // E  in ROOT
      for (int y = 1; y <= n_cx; ++y)      // costh in ROOT
        h.SetBinContent(x, y,
                        static_cast<double>(
                            data[static_cast<size_t>(y - 1) * n_e +
                                 static_cast<size_t>(x - 1)]));
    return h;
  }

  /// Convert to ROOT TH2D with uniform integer binning (0..n_bins).
  /// Useful when bin edges are not available (e.g. rebinned histograms).
  [[nodiscard]] TH2D to_th2d(const char *name = "") const {
    TH2D h(name, "", static_cast<int>(n_e), 0, static_cast<double>(n_e),
           static_cast<int>(n_costh), 0, static_cast<double>(n_costh));
    for (size_t x = 0; x < n_e; ++x)
      for (size_t y = 0; y < n_costh; ++y)
        h.SetBinContent(static_cast<int>(x) + 1, static_cast<int>(y) + 1,
                        static_cast<double>(data[y * n_e + x]));
    return h;
  }
};

/// POD 1D "histogram" — just a flat vector in energy.
using PodHist1D = std::vector<oscillaton_calc_precision>;

/// Convert ROOT TH1D to PodHist1D by copying bin contents.
inline PodHist1D th1d_to_pod(const TH1D &h) {
  PodHist1D ret(static_cast<size_t>(h.GetNbinsX()));
  for (int i = 0; i < h.GetNbinsX(); ++i)
    ret[static_cast<size_t>(i)] =
        static_cast<oscillaton_calc_precision>(h.GetBinContent(i + 1));
  return ret;
}

// ── Poisson chi2 on POD data ─────────────────────────────────────────────

/// Poisson log-likelihood chi2: 2 * Σ [pred − data + data·ln(data/pred)]
/// Returns 0 when data == pred everywhere.
/// Data and prediction may have different underlying types (e.g. double vs float).
template <typename TD, typename TP>
double pod_chi2(const PodHist2D<TD> &data, const PodHist2D<TP> &pred) {
  assert(data.size() == pred.size());
  const size_t n = data.size();
  double chi2 = 0;
#pragma omp parallel for reduction(+ : chi2)
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(data.data[i]);
    const double p = static_cast<double>(pred.data[i]);
    if (d != 0)
      chi2 += (p - d) + d * std::log(d / p);
    else
      chi2 += p;
  }
  return 2 * chi2;
}
