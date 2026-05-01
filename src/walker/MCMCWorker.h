#pragma once

#include "mcmc_concepts.h"
#include <TRandom3.h>
#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>

namespace walker {

template <mcmc_concepts::MCMCState State>
class MCMCWorker {
public:
  std::unique_ptr<State> current;
  std::unique_ptr<State> pending;
  double current_llh{};
  double temperature_ = 1.0;

  // ── Construction ──────────────────────────────────────────────────

  MCMCWorker() noexcept(std::is_nothrow_default_constructible_v<State>)
    requires std::is_default_constructible_v<State>
      : current(std::make_unique<State>()),
        pending(std::make_unique<State>()) {}

  /// ModelAndData-style: GetLogLikelihood() already includes data LLH.
  explicit MCMCWorker(const State &initial, double temperature = 1.0)
      : current(std::make_unique<State>(initial)),
        pending(std::make_unique<State>(initial)),
        current_llh(initial.GetLogLikelihood()),
        temperature_(temperature) {}

  /// Binned-style: prior + data LLH.
  template <typename Data>
  MCMCWorker(const State &initial, const Data &data, double temperature = 1.0)
      : current(std::make_unique<State>(initial)),
        pending(std::make_unique<State>(initial)),
        current_llh(initial.GetLogLikelihood()
                  + initial.GetLogLikelihoodAgainstData(data)),
        temperature_(temperature) {}

  MCMCWorker(const MCMCWorker &)            = delete;
  MCMCWorker &operator=(const MCMCWorker &) = delete;
  MCMCWorker(MCMCWorker &&) noexcept        = default;
  MCMCWorker &operator=(MCMCWorker &&) noexcept = default;

  // ── Step without data (ModelAndData pattern) ──────────────────────

  bool step() {
    *pending = *current;
    pending->proposeStep();
    return accept_(pending->GetLogLikelihood());
  }

  template <typename UniformRNG>
  bool step_with(UniformRNG &&urand) {
    *pending = *current;
    pending->proposeStep();
    return accept_(pending->GetLogLikelihood(),
                   std::forward<UniformRNG>(urand));
  }

  // ── Step with external data (binned pattern) ──────────────────────

  template <typename Data>
  bool step(const Data &data) {
    *pending = *current;
    pending->proposeStep();
    return accept_(pending->GetLogLikelihood()
                 + pending->GetLogLikelihoodAgainstData(data));
  }

  template <typename Data, typename UniformRNG>
  bool step_with(const Data &data, UniformRNG &&urand) {
    *pending = *current;
    pending->proposeStep();
    return accept_(pending->GetLogLikelihood()
                 + pending->GetLogLikelihoodAgainstData(data),
                   std::forward<UniformRNG>(urand));
  }

  // ── Parallel Tempering ────────────────────────────────────────────

  void swap_current(MCMCWorker &other) noexcept {
    current.swap(other.current);
    std::swap(current_llh, other.current_llh);
  }

  // ── Accessors ─────────────────────────────────────────────────────

  [[nodiscard]] State       & state() noexcept { return *current; }
  [[nodiscard]] const State & state() const noexcept { return *current; }

  [[nodiscard]] double temperature() const { return temperature_; }
  void set_temperature(double T) { temperature_ = T; }

  [[nodiscard]] double get_current_llh() const { return current_llh; }

  [[nodiscard]] double acceptance_rate() const {
    return step_count_ > 0
               ? static_cast<double>(accept_count_) /
                     static_cast<double>(step_count_)
               : 0.0;
  }

private:
  bool accept_(double nxt_llh) {
    step_count_++;
    if (nxt_llh > current_llh ||
        gRandom->Rndm() < std::exp((nxt_llh - current_llh) / temperature_)) {
      current.swap(pending);
      current_llh = nxt_llh;
      accept_count_++;
      return true;
    }
    return false;
  }

  template <typename UniformRNG>
  bool accept_(double nxt_llh, UniformRNG &&urand) {
    step_count_++;
    if (nxt_llh > current_llh ||
        urand() < std::exp((nxt_llh - current_llh) / temperature_)) {
      current.swap(pending);
      current_llh = nxt_llh;
      accept_count_++;
      return true;
    }
    return false;
  }

  size_t step_count_ = 0;
  size_t accept_count_ = 0;
};

} // namespace walker
