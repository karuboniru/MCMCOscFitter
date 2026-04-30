#pragma once
#include "pod_hist.hpp"
#include <TH2D.h>
#include <optional>

class SimpleDataHist {
public:
  SimpleDataHist() = default;
  SimpleDataHist &operator=(const SimpleDataHist &) = default;
  SimpleDataHist &operator=(SimpleDataHist &&) = default;
  SimpleDataHist(const SimpleDataHist &) = default;
  SimpleDataHist(SimpleDataHist &&) = default;
  ~SimpleDataHist() = default;

  void proposeStep() {}
  [[nodiscard]] double GetLogLikelihood() const { return 0; }

  void SaveAs(const char *filename) const;
  void LoadFrom(const char *filename);
  void Round();

  // ── TH2D storage (primary until Phase 7) ──────────────────────────────
  TH2D hist_numu, hist_nue, hist_numubar, hist_nuebar;
  std::optional<TH2D> hist_nc;

  // ── POD cache (lazily synced from TH2D) ───────────────────────────────
  [[nodiscard]] const PodHist2D<double> &pod_numu()    const { ensure_pod(); return data_numu; }
  [[nodiscard]] const PodHist2D<double> &pod_numubar() const { ensure_pod(); return data_numubar; }
  [[nodiscard]] const PodHist2D<double> &pod_nue()     const { ensure_pod(); return data_nue; }
  [[nodiscard]] const PodHist2D<double> &pod_nuebar()  const { ensure_pod(); return data_nuebar; }

private:
  void ensure_pod() const;

  mutable bool pod_valid{false};
  mutable PodHist2D<double> data_numu, data_nue, data_numubar, data_nuebar;
};
