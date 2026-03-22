#include <catch2/catch_test_macros.hpp>

#include "walker.h"

#include <TRandom3.h>
#include <cmath>

// Minimal StateI implementation for testing the walker.
struct FixedLLH : public StateI {
  double llh;
  explicit FixedLLH(double l) : llh(l) {}
  void proposeStep() override {}
  double GetLogLikelihood() const override { return llh; }
};

TEST_CASE("MCMCAcceptState", "[walker]") {
  // Use a fixed seed so results are reproducible.
  gRandom->SetSeed(42);

  SECTION("always accepts when next LLH > current LLH") {
    FixedLLH current{-10.0};
    FixedLLH next{-5.0}; // higher log-likelihood → always accept
    for (int i = 0; i < 100; ++i)
      REQUIRE(MCMCAcceptState(current, next));
  }

  SECTION("always accepts equal likelihood") {
    // log_ratio = 0  →  exp(0) = 1  →  rand < 1 always true
    FixedLLH current{-7.0};
    FixedLLH next{-7.0};
    for (int i = 0; i < 100; ++i)
      REQUIRE(MCMCAcceptState(current, next));
  }

  SECTION("never accepts when next LLH << current LLH") {
    // exp(-1000) ≈ 0: acceptance probability is negligible.
    FixedLLH current{0.0};
    FixedLLH next{-1000.0};
    int accepted = 0;
    for (int i = 0; i < 1000; ++i)
      if (MCMCAcceptState(current, next))
        ++accepted;
    REQUIRE(accepted == 0);
  }

  SECTION("accepts with ~expected probability for moderate step-down") {
    // log_ratio = -1  →  acceptance prob = exp(-1) ≈ 0.368
    gRandom->SetSeed(12345);
    FixedLLH current{0.0};
    FixedLLH next{-1.0};
    int accepted = 0;
    constexpr int N = 10000;
    for (int i = 0; i < N; ++i)
      if (MCMCAcceptState(current, next))
        ++accepted;
    double rate = double(accepted) / N;
    REQUIRE(rate > 0.33);
    REQUIRE(rate < 0.41);
  }
}
