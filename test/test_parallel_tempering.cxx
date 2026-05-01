#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "MCMCWorker.h"
#include "ParallelTempering.h"
#include "temperature_ladder.h"
#include "walker.h"

#include <TRandom3.h>
#include <cmath>

// Minimal mock satisfying StateI and MCMCState concept.
// Uses a different name from test_walker.cxx to avoid ODR violations
// (both compile into the same test_physics binary).
struct FixedLLH2 : public StateI {
  double llh;

  explicit FixedLLH2(double l = 0.0) : llh(l) {}
  void proposeStep() override {}
  double GetLogLikelihood() const override { return llh; }
};

// Mock that counts proposal calls — used for step-counting tests.
struct CountingLLH : public StateI {
  double llh = 0.0;
  int propose_count = 0;

  void proposeStep() override { propose_count++; }
  double GetLogLikelihood() const override { return llh; }
};

// ──────────────────────────────────────────────────────────────────────────────
// TemperatureLadder
// ──────────────────────────────────────────────────────────────────────────────

TEST_CASE("TemperatureLadder::geometric", "[temperature_ladder]") {
  SECTION("single chain yields [1.0]") {
    auto ladder = mcmc::TemperatureLadder::geometric(1, 1.0, 100.0);
    REQUIRE(ladder.size() == 1);
    REQUIRE(ladder[0] == 1.0);
    REQUIRE(ladder.cold() == 1.0);
    REQUIRE(ladder.hottest() == 1.0);
  }

  SECTION("two chains: [1.0, T_max]") {
    auto ladder = mcmc::TemperatureLadder::geometric(2, 1.0, 100.0);
    REQUIRE(ladder.size() == 2);
    REQUIRE(ladder[0] == 1.0);
    REQUIRE(ladder[1] == Catch::Approx(100.0));
  }

  SECTION("three chains geometrically spaced") {
    auto ladder = mcmc::TemperatureLadder::geometric(3, 1.0, 100.0);
    REQUIRE(ladder.size() == 3);
    REQUIRE(ladder[0] == 1.0);
    REQUIRE(ladder[2] == 100.0);
    double ratio = ladder[1] / ladder[0];
    REQUIRE(ratio == Catch::Approx(ladder[2] / ladder[1]).epsilon(1e-9));
  }
}

TEST_CASE("TemperatureLadder::beta", "[temperature_ladder]") {
  SECTION("T=1 gives beta=1") {
    auto ladder = mcmc::TemperatureLadder::geometric(1);
    REQUIRE(ladder.beta(0) == 1.0);
  }
  SECTION("T=2 gives beta=0.5") {
    std::vector<double> temps = {1.0, 2.0};
    mcmc::TemperatureLadder ladder(temps);
    REQUIRE(ladder.beta(0) == 1.0);
    REQUIRE(ladder.beta(1) == 0.5);
  }
}

TEST_CASE("TemperatureLadder validation", "[temperature_ladder]") {
  SECTION("empty vector throws") {
    REQUIRE_THROWS(mcmc::TemperatureLadder(std::vector<double>{}));
  }
  SECTION("first not 1.0 throws") {
    REQUIRE_THROWS(mcmc::TemperatureLadder({2.0, 3.0}));
  }
  SECTION("non-strictly-increasing throws") {
    REQUIRE_THROWS(mcmc::TemperatureLadder({1.0, 1.0}));
  }
  SECTION("decreasing order throws") {
    REQUIRE_THROWS(mcmc::TemperatureLadder({1.0, 10.0, 5.0}));
  }
}

TEST_CASE("geometric_ladder free function", "[temperature_ladder]") {
  auto ladder = mcmc::geometric_ladder(5, 1.0, 16.0);
  REQUIRE(ladder.size() == 5);
  REQUIRE(ladder[0] == 1.0);
  REQUIRE(ladder[4] == 16.0);
  // T_5 = T_1 * r^4, so r = 16^(1/4) = 2
  REQUIRE(ladder[1] == Catch::Approx(2.0));
  REQUIRE(ladder[2] == Catch::Approx(4.0));
  REQUIRE(ladder[3] == Catch::Approx(8.0));
}

// ──────────────────────────────────────────────────────────────────────────────
// MCMCAcceptState with temperature
// ──────────────────────────────────────────────────────────────────────────────

TEST_CASE("MCMCAcceptState with temperature", "[walker_temperature]") {
  SECTION("T=1: same as untempered (down-step acceptance ≈ exp(-1))") {
    gRandom->SetSeed(12345);
    FixedLLH2 current{0.0};
    FixedLLH2 next{-1.0};
    int accepted = 0;
    constexpr int N = 10000;
    for (int i = 0; i < N; ++i)
      if (MCMCAcceptState(current, next, 1.0)) ++accepted;
    double rate = static_cast<double>(accepted) / N;
    REQUIRE(rate > 0.33);
    REQUIRE(rate < 0.41);
  }

  SECTION("T=∞: always accepts") {
    gRandom->SetSeed(42);
    FixedLLH2 current{0.0};
    FixedLLH2 next{-1000.0};  // huge penalty, but T=∞ → always accept
    for (int i = 0; i < 100; ++i)
      REQUIRE(MCMCAcceptState(current, next, 1e9));
  }

  SECTION("T=2: higher acceptance than T=1 for down-step") {
    gRandom->SetSeed(99);
    FixedLLH2 current{0.0};
    FixedLLH2 next{-2.0};  // log_ratio = -2
    // T=1: P_accept = exp(-2) ≈ 0.135
    // T=2: P_accept = exp(-1) ≈ 0.368
    int accepted_T1 = 0, accepted_T2 = 0;
    constexpr int N = 10000;
    for (int i = 0; i < N; ++i) {
      if (MCMCAcceptState(current, next, 1.0)) accepted_T1++;
      if (MCMCAcceptState(current, next, 2.0)) accepted_T2++;
    }
    double rate_T1 = static_cast<double>(accepted_T1) / N;
    double rate_T2 = static_cast<double>(accepted_T2) / N;
    REQUIRE(rate_T2 > rate_T1);
    REQUIRE(rate_T2 > 0.30);  // roughly exp(-1)
    REQUIRE(rate_T2 < 0.43);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// MCMCWorker temperature-aware acceptance
// ──────────────────────────────────────────────────────────────────────────────

TEST_CASE("MCMCWorker temperature", "[mcmc_worker_temperature]") {
  SECTION("temperature defaults to 1.0") {
    FixedLLH2 state{0.0};
    walker::MCMCWorker<FixedLLH2> worker(state);
    REQUIRE(worker.temperature() == 1.0);
  }

  SECTION("constructor sets temperature") {
    FixedLLH2 state{0.0};
    walker::MCMCWorker<FixedLLH2> worker(state, 4.5);
    REQUIRE(worker.temperature() == 4.5);
  }

  SECTION("set_temperature works") {
    FixedLLH2 state{0.0};
    walker::MCMCWorker<FixedLLH2> worker(state);
    worker.set_temperature(10.0);
    REQUIRE(worker.temperature() == 10.0);
  }

  SECTION("acceptance rate higher at higher temperature") {
    gRandom->SetSeed(77);
    // Use a state that proposes a worse LLH on every step.
    struct WorseningState : public StateI {
      double llh = 0.0;
      void proposeStep() override {
        llh -= 1.0;  // each proposal is worse by 1.0
      }
      double GetLogLikelihood() const override { return llh; }
    };

    WorseningState s1;
    auto w1 = walker::MCMCWorker<WorseningState>(s1, 1.0);
    int accepted_T1 = 0;
    for (int i = 0; i < 1000; ++i)
      if (w1.step()) accepted_T1++;

    WorseningState s2;
    auto w2 = walker::MCMCWorker<WorseningState>(s2, 100.0);
    int accepted_T2 = 0;
    for (int i = 0; i < 1000; ++i)
      if (w2.step()) accepted_T2++;

    REQUIRE(accepted_T2 > accepted_T1);
  }

  SECTION("get_current_llh exposes cached LLH") {
    FixedLLH2 state{42.0};
    walker::MCMCWorker<FixedLLH2> worker(state);
    REQUIRE(worker.get_current_llh() == 42.0);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// swap_current
// ──────────────────────────────────────────────────────────────────────────────

TEST_CASE("swap_current", "[swap_current]") {
  SECTION("swaps state and cached LLH") {
    FixedLLH2 a_state{10.0};
    FixedLLH2 b_state{20.0};
    walker::MCMCWorker<FixedLLH2> worker_a(a_state);
    walker::MCMCWorker<FixedLLH2> worker_b(b_state);

    REQUIRE(worker_a.get_current_llh() == 10.0);
    REQUIRE(worker_b.get_current_llh() == 20.0);

    worker_a.swap_current(worker_b);

    REQUIRE(worker_a.get_current_llh() == 20.0);
    REQUIRE(worker_b.get_current_llh() == 10.0);
  }

  SECTION("swap does not change temperatures") {
    FixedLLH2 state{0.0};
    walker::MCMCWorker<FixedLLH2> worker_a(state, 1.0);
    walker::MCMCWorker<FixedLLH2> worker_b(state, 10.0);

    worker_a.swap_current(worker_b);

    // Temperatures stay with their original chains.
    REQUIRE(worker_a.temperature() == 1.0);
    REQUIRE(worker_b.temperature() == 10.0);
  }

  SECTION("step after swap uses the new (swapped-in) state") {
    struct CountingState : public StateI {
      int id;
      explicit CountingState(int i) : id(i) {}
      void proposeStep() override { id += 100; }
      double GetLogLikelihood() const override {
        return static_cast<double>(id);
      }
    };

    CountingState s1{1}, s2{2};
    walker::MCMCWorker<CountingState> w1(s1), w2(s2);

    w1.swap_current(w2);
    REQUIRE(w1.state().id == 2);
    REQUIRE(w1.get_current_llh() == 2.0);

    // Taking a step on w1 copies state into pending, proposes, and
    // evaluates acceptance with w1's temperature.
    w1.step();
    // State id was incremented by 100 during proposal.
    REQUIRE(w1.state().id == 102);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// ParallelTempering
// ──────────────────────────────────────────────────────────────────────────────

TEST_CASE("ParallelTempering construction", "[parallel_tempering]") {
  SECTION("ModelAndData pattern") {
    FixedLLH2 initial{5.0};
    auto ladder = mcmc::TemperatureLadder::geometric(4, 1.0, 8.0);
    walker::ParallelTempering<FixedLLH2> pt(ladder, initial);

    REQUIRE(pt.num_chains() == 4);
    REQUIRE(pt.cold_chain().temperature() == 1.0);
    REQUIRE(pt.chain(3).temperature() == 8.0);
    REQUIRE(pt.cold_state().GetLogLikelihood() == 5.0);
  }

  SECTION("single-chain PT (degenerate to standard MCMC)") {
    FixedLLH2 initial{3.0};
    auto ladder = mcmc::TemperatureLadder::geometric(1);
    walker::ParallelTempering<FixedLLH2> pt(ladder, initial);
    REQUIRE(pt.num_chains() == 1);
    REQUIRE(pt.temperatures().cold() == 1.0);

    // step should work without crashing
    for (int i = 0; i < 10; ++i) pt.step();
  }

  SECTION("swap_interval configuration") {
    FixedLLH2 initial{0.0};
    auto ladder = mcmc::TemperatureLadder::geometric(2);
    walker::ParallelTempering<FixedLLH2> pt(ladder, initial);
    REQUIRE(pt.swap_interval() == 100);

    pt.set_swap_interval(50);
    REQUIRE(pt.swap_interval() == 50);
  }
}

TEST_CASE("ParallelTempering step and swap", "[parallel_tempering]") {
  SECTION("stepping advances all chains") {
    CountingLLH initial;
    auto ladder = mcmc::TemperatureLadder::geometric(3, 1.0, 10.0);
    walker::ParallelTempering<CountingLLH> pt(ladder, initial);

    for (int i = 0; i < 5; ++i) pt.step();
    for (size_t i = 0; i < pt.num_chains(); ++i) {
      REQUIRE(pt.chain(i).state().propose_count == 5);
    }
  }

  SECTION("run with callback collects cold chain samples") {
    struct TrackingState : public StateI {
      double val = 0.0;
      TrackingState() = default;
      explicit TrackingState(double v) : val(v) {}
      void proposeStep() override { val += 0.1; }
      double GetLogLikelihood() const override { return val; }
    };

    TrackingState initial{0.0};
    auto ladder = mcmc::TemperatureLadder::geometric(2, 1.0, 100.0);
    walker::ParallelTempering<TrackingState> pt(ladder, initial);
    pt.set_swap_interval(1000);  // no swaps during this short test

    std::vector<double> collected;
    pt.run(5, [&](const TrackingState &s, size_t) {
      collected.push_back(s.val);
    }, 1);

    REQUIRE(collected.size() == 5);
    // Cold chain value should be positive after 5 steps
    REQUIRE(collected.back() > 0.0);
  }
}

TEST_CASE("ParallelTempering swap rate", "[parallel_tempering]") {
  SECTION("same LLH on adjacent chains → swap always accepted") {
    gRandom->SetSeed(111);
    FixedLLH2 initial{0.0};  // proposeStep does nothing, LLH stays 0
    auto ladder = mcmc::TemperatureLadder::geometric(4, 1.0, 100.0);
    walker::ParallelTempering<FixedLLH2> pt(ladder, initial);
    pt.set_swap_interval(1);

    for (int i = 0; i < 100; ++i) pt.step();

    // Same LLH, so every swap attempt should be accepted.
    for (size_t i = 0; i < pt.num_chains() - 1; ++i) {
      REQUIRE(pt.swap_acceptance_rate(i) == 1.0);
    }
  }
}
