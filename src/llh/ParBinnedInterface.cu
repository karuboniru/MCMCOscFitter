#include "ParBinnedInterface.h"
#include "ParBinned.cuh"


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
