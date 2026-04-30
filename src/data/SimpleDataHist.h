#pragma once
#include "pod_hist.hpp"
#include <TH2D.h>
#include <optional>
#include <vector>

class SimpleDataHist {
public:
  SimpleDataHist() = default;
  SimpleDataHist(const SimpleDataHist &) = default;
  SimpleDataHist(SimpleDataHist &&) = default;
  SimpleDataHist &operator=(const SimpleDataHist &) = default;
  SimpleDataHist &operator=(SimpleDataHist &&) = default;
  ~SimpleDataHist() = default;

  void proposeStep() {}
  [[nodiscard]] double GetLogLikelihood() const { return 0; }

  // ── I/O ───────────────────────────────────────────────────────────────
  void SaveAs(const char *filename) const;
  void LoadFrom(const char *filename);
  void Round();

  // ── POD primary storage ───────────────────────────────────────────────
  PodHist2D<double> data_numu, data_nue, data_numubar, data_nuebar;
  std::optional<PodHist2D<double>> data_nc;

  // Bin edges for TH2D conversion (set after LoadFrom / GenerateData).
  std::vector<double> Ebins, costheta_bins;

  // ── TH2D accessors (construct on-the-fly for plotting / I/O) ─────────
  [[nodiscard]] TH2D hist_numu()    const { return data_numu.to_th2d(Ebins, costheta_bins); }
  [[nodiscard]] TH2D hist_numubar() const { return data_numubar.to_th2d(Ebins, costheta_bins); }
  [[nodiscard]] TH2D hist_nue()     const { return data_nue.to_th2d(Ebins, costheta_bins); }
  [[nodiscard]] TH2D hist_nuebar()  const { return data_nuebar.to_th2d(Ebins, costheta_bins); }

  // ── Convenience ───────────────────────────────────────────────────────
  [[nodiscard]] double total_numu()    const;
  [[nodiscard]] double total_numubar() const;
  [[nodiscard]] double total_nue()     const;
  [[nodiscard]] double total_nuebar()  const;
};
