#pragma once

#include "ModelDataLLH.h"
#include <memory>
// #include "OscillationParameters.h"
#include "SimpleDataHist.h"


class ParBinned;
struct param;
class pull_toggle;

class ParBinnedInterface : public ModelDataLLH {
public:
  ParBinnedInterface(std::vector<double> Ebins,
                     std::vector<double> costheta_bins, double scale_ = 1.,
                     size_t E_rebin_factor = 1, size_t costh_rebin_factor = 1,
                     double IH_Bias = 1.0);

  ParBinnedInterface(const ParBinnedInterface &other);
  ParBinnedInterface(ParBinnedInterface &&other) noexcept;
  ParBinnedInterface &operator=(const ParBinnedInterface &other);
  ParBinnedInterface &operator=(ParBinnedInterface &&other) noexcept;
  ~ParBinnedInterface() override;

  void proposeStep() final;

  // virtual double GetLogLikelihood() const override;
  [[nodiscard]] double
  GetLogLikelihoodAgainstData(const StateI &dataset) const final;

  [[nodiscard]] SimpleDataHist GenerateData() const;
  [[nodiscard]] SimpleDataHist GenerateData_NoOsc() const;

  void Print() const;

  void flip_hierarchy();

  void Save_prob_hist(const std::string &name);

  [[nodiscard]] double GetLogLikelihood() const final;

  void UpdatePrediction();

  void SaveAs(const char *filename) const;

  double GetDM32sq() const;
  double GetDM21sq() const;
  double GetT12() const;
  double GetT13() const;
  double GetT23() const;
  double GetDeltaCP() const;

  void set_param(const param &p);

  void set_toggle(const pull_toggle &toggle);
  [[nodiscard]] const pull_toggle & get_toggle() const;

private:
  std::unique_ptr<ParBinned> pImpl;
};