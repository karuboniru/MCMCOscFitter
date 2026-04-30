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
  double                  current_llh{};

  // ── Construction ──────────────────────────────────────────────────

  MCMCWorker() noexcept(std::is_nothrow_default_constructible_v<State>)
    requires std::is_default_constructible_v<State>
      : current(std::make_unique<State>()),
        pending(std::make_unique<State>()) {}

  /// For ModelAndData-style: GetLogLikelihood() already includes data LLH.
  explicit MCMCWorker(const State &initial)
      : current(std::make_unique<State>(initial)),
        pending(std::make_unique<State>(initial)),
        current_llh(initial.GetLogLikelihood()) {}

  /// For binned-style: prior + data LLH.
  template <typename Data>
  MCMCWorker(const State &initial, const Data &data)
      : current(std::make_unique<State>(initial)),
        pending(std::make_unique<State>(initial)),
        current_llh(initial.GetLogLikelihood()
                  + initial.GetLogLikelihoodAgainstData(data)) {}

  MCMCWorker(const MCMCWorker &)            = delete;
  MCMCWorker &operator=(const MCMCWorker &) = delete;
  MCMCWorker(MCMCWorker &&) noexcept        = default;
  MCMCWorker &operator=(MCMCWorker &&) noexcept = default;

  // ── Step without data (ModelAndData pattern) ──────────────────────

  /// Uses gRandom.
  bool step() {
    *pending = *current;
    pending->proposeStep();
    return accept_(pending->GetLogLikelihood());
  }

  /// Custom uniform RNG (callable returning double in [0,1)).
  template <typename UniformRNG>
  bool step_with(UniformRNG &&urand) {
    *pending = *current;
    pending->proposeStep();
    return accept_(pending->GetLogLikelihood(),
                   std::forward<UniformRNG>(urand));
  }

  // ── Step with external data (binned pattern) ──────────────────────

  /// Uses gRandom.
  template <typename Data>
  bool step(const Data &data) {
    *pending = *current;
    pending->proposeStep();
    return accept_(pending->GetLogLikelihood()
                 + pending->GetLogLikelihoodAgainstData(data));
  }

  /// Custom uniform RNG (callable returning double in [0,1)).
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

private:
  bool accept_(double nxt_llh) {
    if (nxt_llh > current_llh ||
        gRandom->Rndm() < std::exp(nxt_llh - current_llh)) {
      current.swap(pending);
      current_llh = nxt_llh;
      return true;
    }
    return false;
  }

  template <typename UniformRNG>
  bool accept_(double nxt_llh, UniformRNG &&urand) {
    if (nxt_llh > current_llh ||
        urand() < std::exp(nxt_llh - current_llh)) {
      current.swap(pending);
      current_llh = nxt_llh;
      return true;
    }
    return false;
  }
};

} // namespace walker
