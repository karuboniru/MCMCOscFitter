#pragma once

#include "MCMCWorker.h"
#include "temperature_ladder.h"

#include <TRandom3.h>
#include <functional>
#include <vector>

namespace walker {

template <mcmc_concepts::MCMCState State>
class ParallelTempering {
public:
  // ── Construction (ModelAndData pattern) ───────────────────────────

  ParallelTempering(const mcmc::TemperatureLadder &ladder,
                    const State &initial)
      : temperatures_(ladder) {
    chains_.reserve(ladder.size());
    for (size_t i = 0; i < ladder.size(); ++i) {
      chains_.emplace_back(initial, ladder[i]);
    }
    swap_accepts_.resize(ladder.size() > 0 ? ladder.size() - 1 : 0, 0);
    swap_attempts_.resize(ladder.size() > 0 ? ladder.size() - 1 : 0, 0);
  }

  // ── Construction (binned pattern) ─────────────────────────────────

  template <typename Data>
  ParallelTempering(const mcmc::TemperatureLadder &ladder,
                    const State &initial, const Data &data)
      : temperatures_(ladder) {
    chains_.reserve(ladder.size());
    for (size_t i = 0; i < ladder.size(); ++i) {
      chains_.emplace_back(initial, data, ladder[i]);
    }
    swap_accepts_.resize(ladder.size() > 0 ? ladder.size() - 1 : 0, 0);
    swap_attempts_.resize(ladder.size() > 0 ? ladder.size() - 1 : 0, 0);
  }

  // ── Step (ModelAndData pattern) ───────────────────────────────────

  void step() {
    for (auto &chain : chains_)
      chain.step();
    step_count_++;
    if (step_count_ % swap_interval_ == 0)
      attempt_swaps();
  }

  // ── Step (binned pattern) ─────────────────────────────────────────

  template <typename Data>
  void step(const Data &data) {
    for (auto &chain : chains_)
      chain.step(data);
    step_count_++;
    if (step_count_ % swap_interval_ == 0)
      attempt_swaps();
  }

  // ── Warmup ────────────────────────────────────────────────────────

  void warmup(size_t n_steps) {
    for (size_t s = 0; s < n_steps; ++s) step();
  }

  template <typename Data>
  void warmup(const Data &data, size_t n_steps) {
    for (size_t s = 0; s < n_steps; ++s) step(data);
  }

  // ── Run with callback ─────────────────────────────────────────────

  void run(size_t iterations,
           std::function<void(const State &, size_t)> on_sample,
           size_t record_interval = 1) {
    for (size_t i = 0; i < iterations; ++i) {
      step();
      if ((i + 1) % record_interval == 0)
        on_sample(cold_state(), step_count_);
    }
  }

  template <typename Data>
  void run(const Data &data, size_t iterations,
           std::function<void(const State &, size_t)> on_sample,
           size_t record_interval = 1) {
    for (size_t i = 0; i < iterations; ++i) {
      step(data);
      if ((i + 1) % record_interval == 0)
        on_sample(cold_state(), step_count_);
    }
  }

  // ── Access ────────────────────────────────────────────────────────

  [[nodiscard]] MCMCWorker<State> &cold_chain() { return chains_.front(); }
  [[nodiscard]] const MCMCWorker<State> &cold_chain() const {
    return chains_.front();
  }
  [[nodiscard]] const State &cold_state() const {
    return cold_chain().state();
  }
  [[nodiscard]] State &cold_state() { return cold_chain().state(); }

  [[nodiscard]] size_t num_chains() const { return chains_.size(); }
  [[nodiscard]] const mcmc::TemperatureLadder &temperatures() const {
    return temperatures_;
  }

  [[nodiscard]] MCMCWorker<State> &chain(size_t i) { return chains_[i]; }
  [[nodiscard]] const MCMCWorker<State> &chain(size_t i) const {
    return chains_[i];
  }

  [[nodiscard]] size_t swap_interval() const { return swap_interval_; }
  void set_swap_interval(size_t n) { swap_interval_ = n > 0 ? n : 1; }

  [[nodiscard]] size_t step_count() const { return step_count_; }

  [[nodiscard]] double swap_acceptance_rate(size_t pair_idx) const {
    if (pair_idx >= swap_attempts_.size() || swap_attempts_[pair_idx] == 0)
      return 0.0;
    return static_cast<double>(swap_accepts_[pair_idx]) /
           static_cast<double>(swap_attempts_[pair_idx]);
  }

  [[nodiscard]] std::vector<double> swap_acceptance_rates() const {
    std::vector<double> rates(swap_attempts_.size());
    for (size_t i = 0; i < rates.size(); ++i)
      rates[i] = swap_acceptance_rate(i);
    return rates;
  }

private:
  void attempt_swaps() {
    size_t start = (step_count_ / swap_interval_) % 2 ? 1 : 0;
    for (size_t i = start; i + 1 < chains_.size(); i += 2) {
      size_t j = i + 1;
      double llh_i = chains_[i].get_current_llh();
      double llh_j = chains_[j].get_current_llh();
      double Ti = temperatures_[i];
      double Tj = temperatures_[j];

      double log_alpha = (llh_i - llh_j) * (1.0 / Ti - 1.0 / Tj);
      swap_attempts_[i]++;

      if (log_alpha > 0.0 || gRandom->Rndm() < std::exp(log_alpha)) {
        chains_[i].swap_current(chains_[j]);
        swap_accepts_[i]++;
      }
    }
  }

  std::vector<MCMCWorker<State>> chains_;
  mcmc::TemperatureLadder temperatures_;
  size_t swap_interval_ = 100;
  size_t step_count_ = 0;
  std::vector<size_t> swap_accepts_;
  std::vector<size_t> swap_attempts_;
};

} // namespace walker
