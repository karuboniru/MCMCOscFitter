#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "binning_tool.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("linspace", "[binning]") {
  SECTION("size is correct") {
    auto v = linspace(0.0, 1.0, 5u);
    REQUIRE(v.size() == 5);
  }

  SECTION("endpoints are exact") {
    auto v = linspace(0.0, 1.0, 5u);
    REQUIRE_THAT(v.front(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(v.back(), WithinAbs(1.0, 1e-12));
  }

  SECTION("values are uniformly spaced") {
    auto v = linspace(0.0, 4.0, 5u);
    for (int i = 0; i < 5; ++i)
      REQUIRE_THAT(v[i], WithinAbs(double(i), 1e-12));
  }

  SECTION("single point") {
    auto v = linspace(3.0, 3.0, 1u);
    REQUIRE(v.size() == 1);
    REQUIRE_THAT(v[0], WithinAbs(3.0, 1e-12));
  }

  SECTION("throws on zero divisions") {
    REQUIRE_THROWS_AS(linspace(0.0, 1.0, 0u), std::length_error);
  }
}

TEST_CASE("logspace", "[binning]") {
  SECTION("size is correct") {
    auto v = logspace(1.0, 1000.0, 4u);
    REQUIRE(v.size() == 4);
  }

  SECTION("endpoints are exact") {
    auto v = logspace(0.1, 20.0, 10u);
    REQUIRE_THAT(v.front(), WithinRel(0.1, 1e-6));
    REQUIRE_THAT(v.back(), WithinAbs(20.0, 1e-12));
  }

  SECTION("10x steps for decade spacing") {
    auto v = logspace(1.0, 1000.0, 4u);
    REQUIRE_THAT(v[1], WithinRel(10.0, 1e-6));
    REQUIRE_THAT(v[2], WithinRel(100.0, 1e-6));
  }

  SECTION("throws on zero divisions") {
    REQUIRE_THROWS_AS(logspace(1.0, 10.0, 0u), std::length_error);
  }
}

TEST_CASE("to_center", "[binning]") {
  SECTION("returns N-1 midpoints for N edges") {
    std::vector<double> edges{0.0, 2.0, 4.0, 10.0};
    auto centers = to_center<double>(edges);
    REQUIRE(centers.size() == 3);
    REQUIRE_THAT(centers[0], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(centers[1], WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(centers[2], WithinAbs(7.0, 1e-12));
  }

  SECTION("float precision version") {
    std::vector<double> edges{0.0, 1.0};
    auto centers = to_center<float>(edges);
    REQUIRE(centers.size() == 1);
    REQUIRE_THAT(double(centers[0]), WithinAbs(0.5, 1e-6));
  }
}

TEST_CASE("to_center_g (geometric)", "[binning]") {
  SECTION("geometric mean of edges") {
    std::vector<double> edges{1.0, 4.0};
    auto centers = to_center_g<double>(edges);
    REQUIRE(centers.size() == 1);
    REQUIRE_THAT(centers[0], WithinRel(2.0, 1e-9));
  }
}

TEST_CASE("divide_bins", "[binning]") {
  SECTION("output size is (N-1)*multiplier + 1") {
    std::vector<double> v{0.0, 1.0, 2.0};
    auto d = divide_bins<double>(v, 3u);
    REQUIRE(d.size() == (2 * 3) + 1);
  }

  SECTION("preserves original edges") {
    std::vector<double> v{0.0, 1.0, 2.0};
    auto d = divide_bins<double>(v, 4u);
    REQUIRE_THAT(d.front(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(d.back(), WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(d[4], WithinAbs(1.0, 1e-12));
  }
}

TEST_CASE("divide_bins_log", "[binning]") {
  SECTION("output size is (N-1)*multiplier + 1") {
    std::vector<double> v{1.0, 10.0, 100.0};
    auto d = divide_bins_log<double>(v, 5u);
    REQUIRE(d.size() == (2 * 5) + 1);
  }

  SECTION("preserves original edges") {
    std::vector<double> v{1.0, 10.0, 100.0};
    auto d = divide_bins_log<double>(v, 2u);
    REQUIRE_THAT(d.front(), WithinRel(1.0, 1e-9));
    REQUIRE_THAT(d[2], WithinRel(10.0, 1e-9));
    REQUIRE_THAT(d.back(), WithinRel(100.0, 1e-9));
  }
}
