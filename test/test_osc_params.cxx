#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "OscillationParameters.h"

#include <TRandom3.h>
#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// PDG 2023 central values baked into OscillationParameters.h
static constexpr double kDM2_NH = 2.455e-3;
static constexpr double kDM2_IH = -2.529e-3;
static constexpr double kT23_NH = 0.558;
static constexpr double kT23_IH = 0.553;
static constexpr double kT13 = 2.19e-2;
static constexpr double kDm2 = 7.53e-5;
static constexpr double kT12 = 0.307;
static constexpr double kDCP = 1.19 * M_PI;

TEST_CASE("OscillationParameters default construction (NH)", "[osc_params]") {
  OscillationParameters p;

  SECTION("normal hierarchy by default") { REQUIRE(p.GetDM32sq() > 0); }

  SECTION("PDG central values") {
    REQUIRE_THAT(p.GetDM32sq(), WithinRel(kDM2_NH, 1e-6));
    REQUIRE_THAT(p.GetT23(), WithinRel(kT23_NH, 1e-6));
    REQUIRE_THAT(p.GetT13(), WithinRel(kT13, 1e-6));
    REQUIRE_THAT(p.GetDM21sq(), WithinRel(kDm2, 1e-6));
    REQUIRE_THAT(p.GetT12(), WithinRel(kT12, 1e-6));
    REQUIRE_THAT(p.GetDeltaCP(), WithinAbs(kDCP, 1e-9));
  }
}

TEST_CASE("OscillationParameters::GetLogLikelihood prior", "[osc_params]") {
  OscillationParameters p;

  SECTION("all priors off → zero") {
    p.set_toggle(all_off);
    REQUIRE_THAT(p.GetLogLikelihood(), WithinAbs(0.0, 1e-12));
  }

  SECTION("at central values with all priors on → zero penalty") {
    p.set_toggle(all_on);
    // At PDG central values every Gaussian term is 0.
    REQUIRE_THAT(p.GetLogLikelihood(), WithinAbs(0.0, 1e-9));
  }

  SECTION("explicit pull_toggle: only DM32 active") {
    pull_toggle t{};
    t[0] = true; // DM32
    // At central value penalty is 0.
    REQUIRE_THAT(p.GetLogLikelihood(t), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("OscillationParameters::flip_hierarchy", "[osc_params]") {
  OscillationParameters p;
  REQUIRE(p.GetDM32sq() > 0); // NH

  p.flip_hierarchy();
  REQUIRE(p.GetDM32sq() < 0); // IH
  REQUIRE_THAT(p.GetDM32sq(), WithinRel(kDM2_IH, 1e-6));
  REQUIRE_THAT(p.GetT23(), WithinRel(kT23_IH, 1e-6));

  p.flip_hierarchy();
  REQUIRE(p.GetDM32sq() > 0); // back to NH
  REQUIRE_THAT(p.GetDM32sq(), WithinRel(kDM2_NH, 1e-6));
}

TEST_CASE("OscillationParameters::set_param", "[osc_params]") {
  OscillationParameters p;

  SECTION("positive DM2 sets NH") {
    param q{.DM2 = 2.5e-3, .Dm2 = 7.5e-5, .T23 = 0.55,
            .T13 = 0.022, .T12 = 0.31, .DCP = 1.5};
    p.set_param(q);
    REQUIRE(p.GetDM32sq() > 0);
    REQUIRE_THAT(p.GetDM32sq(), WithinRel(2.5e-3, 1e-9));
    REQUIRE_THAT(p.GetT23(), WithinRel(0.55, 1e-9));
  }

  SECTION("negative DM2 sets IH") {
    param q{.DM2 = -2.5e-3, .Dm2 = 7.5e-5, .T23 = 0.55,
            .T13 = 0.022, .T12 = 0.31, .DCP = 1.5};
    p.set_param(q);
    REQUIRE(p.GetDM32sq() < 0);
    REQUIRE_THAT(p.GetDM32sq(), WithinRel(-2.5e-3, 1e-9));
  }
}

TEST_CASE("OscillationParameters::proposeStep moves parameters", "[osc_params]") {
  gRandom->SetSeed(99);
  OscillationParameters p;
  const double initial_DM2 = p.GetDM32sq();

  // Run several steps; at least one should change the value.
  bool changed = false;
  for (int i = 0; i < 50; ++i) {
    p.proposeStep();
    if (std::abs(p.GetDM32sq()) != std::abs(initial_DM2))
      changed = true;
  }
  REQUIRE(changed);
}

TEST_CASE("pull_toggle helper methods", "[osc_params]") {
  SECTION("all_on activates all 6 parameters") {
    REQUIRE(all_on.get_active().size() == 6);
    REQUIRE(all_on.get_inactive().empty());
  }

  SECTION("all_off deactivates all 6 parameters") {
    REQUIRE(all_off.get_active().empty());
    REQUIRE(all_off.get_inactive().size() == 6);
  }
}
