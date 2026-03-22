#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "constants.h"

using Catch::Matchers::WithinRel;

TEST_CASE("scale_factor consistency", "[constants]") {
  SECTION("6-year factor is 6x the 1-year factor") {
    REQUIRE_THAT(scale_factor_6y, WithinRel(6.0 * scale_factor_1y, 1e-12));
  }

  SECTION("scale factors are positive and non-zero") {
    REQUIRE(scale_factor_1y > 0.0);
    REQUIRE(scale_factor_6y > 0.0);
  }

  SECTION("scale_factor_1y has correct order of magnitude") {
    // atmo_count_C12 * time_1y / 1e42
    // ≈ (2e10/13.6 * 6.02e23) * 3.16e7 / 1e42 ≈ 2.8e-2
    REQUIRE(scale_factor_1y > 1e-3);
    REQUIRE(scale_factor_1y < 1.0);
  }
}

TEST_CASE("H_to_C ratio", "[constants]") {
  SECTION("positive") { REQUIRE(H_to_C > 0.0); }

  SECTION("H_mass_perc=12 gives ~1.6 hydrogen atoms per carbon") {
    // 12% H by mass in CH_x -> H_to_C = (12/1) / (88/12) ≈ 1.636
    REQUIRE_THAT(H_to_C, WithinRel(1.636, 0.01));
  }
}

TEST_CASE("atmo_count_C12 is physically reasonable", "[constants]") {
  // 2e10 g liquid scintillator / ~13 g/mol * Avogadro ~ 9e32 molecules
  REQUIRE(atmo_count_C12 > 1e31);
  REQUIRE(atmo_count_C12 < 1e34);
}
