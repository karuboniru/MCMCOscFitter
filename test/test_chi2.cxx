#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "chi2.h"

#include <TH1.h>
#include <TH2.h>
#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Helpers to build simple 1x1 TH2D histograms with one content bin.
static TH2D make_th2(double content, const char *name = "h") {
  TH2D h(name, "", 1, 0, 1, 1, 0, 1);
  h.SetBinContent(1, 1, content);
  return h;
}

static TH2D make_th2_uniform(int nx, int ny, double content,
                             const char *name = "h") {
  TH2D h(name, "", nx, 0, nx, ny, 0, ny);
  for (int x = 1; x <= nx; ++x)
    for (int y = 1; y <= ny; ++y)
      h.SetBinContent(x, y, content);
  return h;
}

TEST_CASE("TH2D_chi2: data == prediction", "[chi2]") {
  TH1::AddDirectory(false);

  SECTION("single bin, data = pred = 1") {
    auto data = make_th2(1.0, "d1");
    auto pred = make_th2(1.0, "p1");
    // term: (1 - 1) + 1 * ln(1/1) = 0  →  chi2 = 0
    REQUIRE_THAT(TH2D_chi2(data, pred), WithinAbs(0.0, 1e-12));
  }

  SECTION("single bin, data = pred = 5") {
    auto data = make_th2(5.0, "d5");
    auto pred = make_th2(5.0, "p5");
    REQUIRE_THAT(TH2D_chi2(data, pred), WithinAbs(0.0, 1e-10));
  }

  SECTION("uniform 4x4 grid, all equal") {
    auto data = make_th2_uniform(4, 4, 3.0, "du");
    auto pred = make_th2_uniform(4, 4, 3.0, "pu");
    REQUIRE_THAT(TH2D_chi2(data, pred), WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE("TH2D_chi2: data == 0 bins", "[chi2]") {
  TH1::AddDirectory(false);

  SECTION("data=0, pred>0: contributes 2*pred to chi2") {
    auto data = make_th2(0.0, "dz");
    auto pred = make_th2(3.0, "pz");
    // term: pred = 3  →  chi2 = 2 * 3 = 6
    REQUIRE_THAT(TH2D_chi2(data, pred), WithinAbs(6.0, 1e-12));
  }

  SECTION("data=0, pred=0: contributes nothing") {
    auto data = make_th2(0.0, "dz2");
    auto pred = make_th2(0.0, "pz2");
    REQUIRE_THAT(TH2D_chi2(data, pred), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("TH2D_chi2: known analytic values", "[chi2]") {
  TH1::AddDirectory(false);

  SECTION("data=2, pred=4: 2*((4-2)+2*ln(2/4)) = 2*(2 - 2*ln2)") {
    auto data = make_th2(2.0, "da");
    auto pred = make_th2(4.0, "pa");
    double expected = 2.0 * ((4.0 - 2.0) + 2.0 * std::log(2.0 / 4.0));
    REQUIRE_THAT(TH2D_chi2(data, pred), WithinAbs(expected, 1e-10));
  }

  SECTION("chi2 is non-negative for any pred/data > 0") {
    // By convexity of x - 1 - ln(x), the Poisson chi2 is always >= 0.
    for (double d : {0.5, 1.0, 2.0, 5.0, 10.0}) {
      for (double p : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        auto data = make_th2(d, "dn");
        auto pred = make_th2(p, "pn");
        REQUIRE(TH2D_chi2(data, pred) >= 0.0);
      }
    }
  }

  SECTION("chi2 is symmetric in data/pred direction only when d==p") {
    auto data = make_th2(3.0, "ds");
    auto pred = make_th2(5.0, "ps");
    auto data2 = make_th2(5.0, "ds2");
    auto pred2 = make_th2(3.0, "ps2");
    // TH2D_chi2(d,p) != TH2D_chi2(p,d) in general (Poisson, not chi2)
    REQUIRE(TH2D_chi2(data, pred) != TH2D_chi2(data2, pred2));
  }
}

TEST_CASE("TH2D_chi2: additive across bins", "[chi2]") {
  TH1::AddDirectory(false);

  // chi2 of a 2-bin histogram should be sum of individual 1-bin chi2s.
  TH2D data2("d2b", "", 2, 0, 2, 1, 0, 1);
  TH2D pred2("p2b", "", 2, 0, 2, 1, 0, 1);
  data2.SetBinContent(1, 1, 2.0);
  data2.SetBinContent(2, 1, 5.0);
  pred2.SetBinContent(1, 1, 4.0);
  pred2.SetBinContent(2, 1, 5.0);

  auto d1 = make_th2(2.0, "d1b");
  auto p1 = make_th2(4.0, "p1b");
  auto d2 = make_th2(5.0, "d2bb");
  auto p2 = make_th2(5.0, "p2bb");

  double expected = TH2D_chi2(d1, p1) + TH2D_chi2(d2, p2);
  REQUIRE_THAT(TH2D_chi2(data2, pred2), WithinAbs(expected, 1e-10));
}
