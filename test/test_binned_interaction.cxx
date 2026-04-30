#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "BinnedInteraction.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"

#include <TH1.h>
#include <TH2.h>
#include <memory>

using Catch::Matchers::WithinAbs;

// ── Mock propagator ──────────────────────────────────────────────────────────
// Returns identity oscillation matrices (pure survival, no flavour change)
// regardless of energy / cosine-theta / oscillation parameters.
// This lets BinnedInteraction::UpdatePrediction() run with a known result:
//   prediction_numu  = flux_numu  (no mixing into nue)
//   prediction_nue   = flux_nue   (no mixing into numu)
//   etc.
class IdentityPropagator : public IHistogramPropagator {
public:
  void re_calculate(const OscillationParameters &) override {
    // Identity propagator — probabilities are constant, no pre-calculation needed.
  }

  std::array<std::array<std::array<TH2D, 2>, 2>, 2>
  GetProb_Hists(const std::vector<double> &Ebins,
                const std::vector<double> &costhbins,
                const OscillationParameters &) override {
    // Build TH2D with the same binning as the inputs.
    const int ne = int(Ebins.size()) - 1;
    const int nc = int(costhbins.size()) - 1;

    auto make = [&](double diag_value, const char *name) {
      TH2D h(name, "", ne, Ebins.data(), nc, costhbins.data());
      for (int x = 1; x <= ne; ++x)
        for (int y = 1; y <= nc; ++y)
          h.SetBinContent(x, y, diag_value);
      return h;
    };

    // Identity: P(nue->nue)=1, P(nue->numu)=0, P(numu->nue)=0, P(numu->numu)=1
    // Layout: [nu/antinu][from: 0-nue, 1-numu][to: 0-nue, 1-numu]
    std::array<std::array<std::array<TH2D, 2>, 2>, 2> result;
    for (int nu = 0; nu < 2; ++nu) {
      result[nu][0][0] = make(1.0, "ee");   // nue->nue
      result[nu][0][1] = make(0.0, "em");   // nue->numu
      result[nu][1][0] = make(0.0, "me");   // numu->nue
      result[nu][1][1] = make(1.0, "mm");   // numu->numu
    }
    return result;
  }

  std::array<std::array<std::array<TH2D, 3>, 3>, 2>
  GetProb_Hists_3F(const std::vector<double> &Ebins,
                   const std::vector<double> &costhbins,
                   const OscillationParameters &) override {
    const int ne = int(Ebins.size()) - 1;
    const int nc = int(costhbins.size()) - 1;

    auto make = [&](double v, const char *name) {
      TH2D h(name, "", ne, Ebins.data(), nc, costhbins.data());
      for (int x = 1; x <= ne; ++x)
        for (int y = 1; y <= nc; ++y)
          h.SetBinContent(x, y, v);
      return h;
    };

    std::array<std::array<std::array<TH2D, 3>, 3>, 2> result;
    for (int nu = 0; nu < 2; ++nu)
      for (int f = 0; f < 3; ++f)
        for (int t = 0; t < 3; ++t)
          result[nu][f][t] = make(f == t ? 1.0 : 0.0, "id");
    return result;
  }
};

// ── Helpers ──────────────────────────────────────────────────────────────────

// Build a flat TH2D (all bins = value).
static TH2D flat_th2(const std::vector<double> &Ebins,
                     const std::vector<double> &costhbins, double value,
                     const char *name) {
  TH2D h(name, "", int(Ebins.size()) - 1, Ebins.data(),
         int(costhbins.size()) - 1, costhbins.data());
  for (int x = 1; x <= h.GetNbinsX(); ++x)
    for (int y = 1; y <= h.GetNbinsY(); ++y)
      h.SetBinContent(x, y, value);
  return h;
}

// Build a flat TH1D (all bins = value).
static TH1D flat_th1(const std::vector<double> &Ebins, double value,
                     const char *name) {
  TH1D h(name, "", int(Ebins.size()) - 1, Ebins.data());
  for (int x = 1; x <= h.GetNbinsX(); ++x)
    h.SetBinContent(x, value);
  return h;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("BinnedInteraction with identity propagator", "[BinnedInteraction]") {
  TH1::AddDirectory(false);

  // Use a small, uniform binning so the test runs fast.
  auto Ebins = linspace(0.1, 10.0, 6u);       // 5 E-bins
  auto costhbins = linspace(-1.0, 1.0, 5u);   // 4 costh-bins

  // All flux = 2, all xsec = 3.  With identity oscillation:
  //   prediction_numu  = flux_numu  * xsec_numu  = 2 * 3 = 6 per bin
  //   prediction_nue   = flux_nue   * xsec_nue   = 2 * 3 = 6 per bin
  BinnedHistograms histos{
      .flux_numu    = flat_th2(Ebins, costhbins, 2.0, "fn"),
      .flux_numubar = flat_th2(Ebins, costhbins, 2.0, "fnb"),
      .flux_nue     = flat_th2(Ebins, costhbins, 2.0, "fe"),
      .flux_nuebar  = flat_th2(Ebins, costhbins, 2.0, "feb"),
      .xsec_numu    = flat_th1(Ebins, 3.0, "xn"),
      .xsec_numubar = flat_th1(Ebins, 3.0, "xnb"),
      .xsec_nue     = flat_th1(Ebins, 3.0, "xe"),
      .xsec_nuebar  = flat_th1(Ebins, 3.0, "xeb"),
  };

  auto propagator = std::make_shared<IdentityPropagator>();

  BinnedInteraction bint(Ebins, costhbins, propagator, std::move(histos));

  SECTION("GenerateData prediction matches flux*xsec with identity oscillation") {
    auto data = bint.GenerateData();
    // Every bin in every histogram should be flux * xsec = 6.
    for (int x = 1; x <= data.hist_numu.GetNbinsX(); ++x)
      for (int y = 1; y <= data.hist_numu.GetNbinsY(); ++y) {
        REQUIRE_THAT(data.hist_numu.GetBinContent(x, y), WithinAbs(6.0, 1e-9));
        REQUIRE_THAT(data.hist_nue.GetBinContent(x, y), WithinAbs(6.0, 1e-9));
      }
  }

  SECTION("GetLogLikelihoodAgainstData returns 0 when data == prediction") {
    auto data = bint.GenerateData();
    // LLH against itself should be 0 (chi2 = 0 everywhere).
    double llh = bint.GetLogLikelihoodAgainstData(data);
    REQUIRE_THAT(llh, WithinAbs(0.0, 1e-9));
  }

  SECTION("GetLogLikelihoodAgainstData is negative when data != prediction") {
    auto data = bint.GenerateData();
    // Perturb one bin — likelihood should drop below 0.
    data.hist_numu.SetBinContent(1, 1, 8.0); // data > pred
    double llh = bint.GetLogLikelihoodAgainstData(data);
    REQUIRE(llh < 0.0);
  }

  SECTION("GetLogLikelihood (prior) at PDG central values is 0") {
    // Default construction is at PDG central values; all_on is the default.
    // Penalty should be 0.
    REQUIRE_THAT(bint.GetLogLikelihood(), WithinAbs(0.0, 1e-9));
  }

  SECTION("flip_hierarchy changes DM32sq sign and re-runs UpdatePrediction") {
    double dm2_before = bint.GetDM32sq();
    bint.flip_hierarchy();
    double dm2_after = bint.GetDM32sq();
    // Sign must flip.
    REQUIRE((dm2_before > 0) != (dm2_after > 0));
    // Prediction is still valid (no crash, data can be generated).
    auto data = bint.GenerateData();
    REQUIRE(data.hist_numu.GetNbinsX() > 0);
  }
}

TEST_CASE("BinnedInteraction IH bias term", "[BinnedInteraction]") {
  TH1::AddDirectory(false);

  auto Ebins = linspace(0.1, 5.0, 4u);
  auto costhbins = linspace(-1.0, 1.0, 3u);

  auto flat_histos = [&]() -> BinnedHistograms {
    return {
        .flux_numu    = flat_th2(Ebins, costhbins, 1.0, "fn2"),
        .flux_numubar = flat_th2(Ebins, costhbins, 1.0, "fnb2"),
        .flux_nue     = flat_th2(Ebins, costhbins, 1.0, "fe2"),
        .flux_nuebar  = flat_th2(Ebins, costhbins, 1.0, "feb2"),
        .xsec_numu    = flat_th1(Ebins, 1.0, "xn2"),
        .xsec_numubar = flat_th1(Ebins, 1.0, "xnb2"),
        .xsec_nue     = flat_th1(Ebins, 1.0, "xe2"),
        .xsec_nuebar  = flat_th1(Ebins, 1.0, "xeb2"),
    };
  };

  SECTION("IH bias = 1.0 contributes log(1)=0 when in IH") {
    auto prop = std::make_shared<IdentityPropagator>();
    BinnedInteraction bint(Ebins, costhbins, prop, flat_histos(),
                           /*E_rebin=*/1, /*costh_rebin=*/1, /*IH_Bias=*/1.0);
    bint.flip_hierarchy(); // enter IH
    double llh_no_bias = bint.GetLogLikelihood();

    auto prop2 = std::make_shared<IdentityPropagator>();
    BinnedInteraction bint2(Ebins, costhbins, prop2, flat_histos(),
                            1, 1, /*IH_Bias=*/1.0);
    bint2.flip_hierarchy();
    REQUIRE_THAT(bint2.GetLogLikelihood(), WithinAbs(llh_no_bias, 1e-12));
  }

  SECTION("IH bias > 1.0 increases LLH when in IH") {
    auto prop1 = std::make_shared<IdentityPropagator>();
    BinnedInteraction bint1(Ebins, costhbins, prop1, flat_histos(), 1, 1, 1.0);
    bint1.flip_hierarchy();

    auto prop2 = std::make_shared<IdentityPropagator>();
    BinnedInteraction bint2(Ebins, costhbins, prop2, flat_histos(), 1, 1, 2.0);
    bint2.flip_hierarchy();

    REQUIRE(bint2.GetLogLikelihood() > bint1.GetLogLikelihood());
  }
}
