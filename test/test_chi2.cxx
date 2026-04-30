#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "pod_hist.hpp"

#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Helpers to build simple PodHist2D<double> with uniform content.
static PodHist2D<double> make_pod(size_t n_costh, size_t n_e, double content) {
  PodHist2D<double> pod(n_costh, n_e);
  for (size_t c = 0; c < n_costh; ++c)
    for (size_t e = 0; e < n_e; ++e)
      pod(c, e) = content;
  return pod;
}

TEST_CASE("pod_chi2: data == prediction", "[chi2]") {
  SECTION("single bin, data = pred = 1") {
    auto data = make_pod(1, 1, 1.0);
    auto pred = make_pod(1, 1, 1.0);
    REQUIRE_THAT(pod_chi2(data, pred), WithinAbs(0.0, 1e-12));
  }

  SECTION("single bin, data = pred = 5") {
    auto data = make_pod(1, 1, 5.0);
    auto pred = make_pod(1, 1, 5.0);
    REQUIRE_THAT(pod_chi2(data, pred), WithinAbs(0.0, 1e-10));
  }

  SECTION("uniform 4x4 grid, all equal") {
    auto data = make_pod(4, 4, 3.0);
    auto pred = make_pod(4, 4, 3.0);
    REQUIRE_THAT(pod_chi2(data, pred), WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE("pod_chi2: data == 0 bins", "[chi2]") {
  SECTION("data=0, pred>0: contributes 2*pred to chi2") {
    auto data = make_pod(1, 1, 0.0);
    auto pred = make_pod(1, 1, 3.0);
    REQUIRE_THAT(pod_chi2(data, pred), WithinAbs(6.0, 1e-12));
  }

  SECTION("data=0, pred=0: contributes nothing") {
    auto data = make_pod(1, 1, 0.0);
    auto pred = make_pod(1, 1, 0.0);
    REQUIRE_THAT(pod_chi2(data, pred), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("pod_chi2: known analytic values", "[chi2]") {
  SECTION("data=2, pred=4: 2*((4-2)+2*ln(2/4)) = 2*(2 - 2*ln2)") {
    auto data = make_pod(1, 1, 2.0);
    auto pred = make_pod(1, 1, 4.0);
    double expected = 2.0 * ((4.0 - 2.0) + 2.0 * std::log(2.0 / 4.0));
    REQUIRE_THAT(pod_chi2(data, pred), WithinAbs(expected, 1e-10));
  }

  SECTION("chi2 is non-negative for any pred/data > 0") {
    for (double d : {0.5, 1.0, 2.0, 5.0, 10.0}) {
      for (double p : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        auto data = make_pod(1, 1, d);
        auto pred = make_pod(1, 1, p);
        REQUIRE(pod_chi2(data, pred) >= 0.0);
      }
    }
  }

  SECTION("chi2 is symmetric in data/pred direction only when d==p") {
    auto data = make_pod(1, 1, 3.0);
    auto pred = make_pod(1, 1, 5.0);
    auto data2 = make_pod(1, 1, 5.0);
    auto pred2 = make_pod(1, 1, 3.0);
    REQUIRE(pod_chi2(data, pred) != pod_chi2(data2, pred2));
  }
}

TEST_CASE("pod_chi2: additive across bins", "[chi2]") {
  // chi2 of a 2-bin histogram should be sum of individual 1-bin chi2s.
  PodHist2D<double> data2(1, 2);
  PodHist2D<double> pred2(1, 2);
  data2(0, 0) = 2.0;
  data2(0, 1) = 5.0;
  pred2(0, 0) = 4.0;
  pred2(0, 1) = 5.0;

  auto d1 = make_pod(1, 1, 2.0);
  auto p1 = make_pod(1, 1, 4.0);
  auto d2 = make_pod(1, 1, 5.0);
  auto p2 = make_pod(1, 1, 5.0);

  double expected = pod_chi2(d1, p1) + pod_chi2(d2, p2);
  REQUIRE_THAT(pod_chi2(data2, pred2), WithinAbs(expected, 1e-10));
}
