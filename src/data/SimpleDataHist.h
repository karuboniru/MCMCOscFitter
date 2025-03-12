#pragma once
#include "StateI.h"
#include <TH2D.h>
#include <optional>

class SimpleDataHist : virtual public StateI {
public:
  SimpleDataHist() = default;
  SimpleDataHist &operator=(const SimpleDataHist &) = default;
  SimpleDataHist &operator=(SimpleDataHist &&) = default;
  SimpleDataHist(const SimpleDataHist &) = default;
  SimpleDataHist(SimpleDataHist &&) = default;
  ~SimpleDataHist() = default;

  void proposeStep() final {}
  [[nodiscard]] double GetLogLikelihood() const final { return 0; }

  void SaveAs(const char *filename) const;
  void LoadFrom(const char *filename);
  void Round();

  TH2D hist_numu, hist_nue, hist_numubar, hist_nuebar;
  std::optional<TH2D> hist_nc;
};
