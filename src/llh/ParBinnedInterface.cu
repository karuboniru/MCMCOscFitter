#include "ParBinned.cuh"
#include "ParBinnedInterface.h"

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
