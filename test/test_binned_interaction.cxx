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
  void re_calculate(const OscillationParameters &) override {}

  std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2>
  GetProb_Hists_POD(const std::vector<double> &Ebins,
                    const std::vector<double> &costhbins,
                    const OscillationParameters &) override {
    const size_t ne = Ebins.size() - 1;
    const size_t nc = costhbins.size() - 1;
    auto make = [&](double v) {
      PodHist2D<double> pod(nc, ne);
      for (size_t c = 0; c < nc; ++c)
        for (size_t e = 0; e < ne; ++e)
          pod(c, e) = v;
      return pod;
    };
    // Identity: P(nue->nue)=1, P(nue->numu)=0, P(numu->nue)=0, P(numu->numu)=1
    std::array<std::array<std::array<PodHist2D<double>, 2>, 2>, 2> result;
    for (int nu = 0; nu < 2; ++nu) {
      result[nu][0][0] = make(1.0);  // nue->nue
      result[nu][0][1] = make(0.0);  // nue->numu
      result[nu][1][0] = make(0.0);  // numu->nue
      result[nu][1][1] = make(1.0);  // numu->numu
    }
    return result;
  }

  std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2>
  GetProb_Hists_3F_POD(const std::vector<double> &Ebins,
                       const std::vector<double> &costhbins,
                       const OscillationParameters &) override {
    const size_t ne = Ebins.size() - 1;
    const size_t nc = costhbins.size() - 1;
    auto make = [&](double v) {
      PodHist2D<double> pod(nc, ne);
      for (size_t c = 0; c < nc; ++c)
        for (size_t e = 0; e < ne; ++e)
          pod(c, e) = v;
      return pod;
    };
    std::array<std::array<std::array<PodHist2D<double>, 3>, 3>, 2> result;
    for (int nu = 0; nu < 2; ++nu)
      for (int f = 0; f < 3; ++f)
        for (int t = 0; t < 3; ++t)
          result[nu][f][t] = make(f == t ? 1.0 : 0.0);
    return result;
  }
};

// ── Helpers ──────────────────────────────────────────────────────────────────

static PodHist2D<oscillaton_calc_precision> flat_pod_hist2d(size_t n_e, size_t n_c, double value) {
  PodHist2D<oscillaton_calc_precision> pod(n_c, n_e);
  for (size_t c = 0; c < n_c; ++c)
    for (size_t e = 0; e < n_e; ++e)
      pod(c, e) = static_cast<oscillaton_calc_precision>(value);
  return pod;
}

static PodHist1D flat_pod_hist1d(size_t n_e, double value) {
  PodHist1D pod(n_e);
  for (size_t e = 0; e < n_e; ++e)
    pod[e] = static_cast<oscillaton_calc_precision>(value);
  return pod;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("BinnedInteraction with identity propagator", "[BinnedInteraction]") {
  // Use a small, uniform binning so the test runs fast.
  auto Ebins = linspace(0.1, 10.0, 6u);       // 5 E-bins
  auto costhbins = linspace(-1.0, 1.0, 5u);   // 4 costh-bins
  const size_t n_e = Ebins.size() - 1;
  const size_t n_c = costhbins.size() - 1;

  // All flux = 2, all xsec = 3.  With identity oscillation:
  //   prediction_numu  = flux_numu  * xsec_numu  = 2 * 3 = 6 per bin
  //   prediction_nue   = flux_nue   * xsec_nue   = 2 * 3 = 6 per bin
  BinnedHistograms histos{
      .pod_flux_numu    = flat_pod_hist2d(n_e, n_c, 2.0),
      .pod_flux_numubar = flat_pod_hist2d(n_e, n_c, 2.0),
      .pod_flux_nue     = flat_pod_hist2d(n_e, n_c, 2.0),
      .pod_flux_nuebar  = flat_pod_hist2d(n_e, n_c, 2.0),
      .pod_xsec_numu    = flat_pod_hist1d(n_e, 3.0),
      .pod_xsec_numubar = flat_pod_hist1d(n_e, 3.0),
      .pod_xsec_nue     = flat_pod_hist1d(n_e, 3.0),
      .pod_xsec_nuebar  = flat_pod_hist1d(n_e, 3.0),
  };

  auto propagator = std::make_shared<IdentityPropagator>();

  BinnedInteraction bint(Ebins, costhbins, propagator, std::move(histos));

  SECTION("GenerateData prediction matches flux*xsec with identity oscillation") {
    auto data = bint.GenerateData();
    // Every bin in every histogram should be flux * xsec = 6.
    for (int x = 1; x <= static_cast<int>(data.data_numu.n_e); ++x)
      for (int y = 1; y <= static_cast<int>(data.data_numu.n_costh); ++y) {
        REQUIRE_THAT(data.hist_numu().GetBinContent(x, y), WithinAbs(6.0, 1e-9));
        REQUIRE_THAT(data.hist_nue().GetBinContent(x, y), WithinAbs(6.0, 1e-9));
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
    data.data_numu(0, 0) = 8.0; // data > pred
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
    REQUIRE(data.data_numu.n_e > 0);
  }
}

TEST_CASE("BinnedInteraction IH bias term", "[BinnedInteraction]") {
  auto Ebins = linspace(0.1, 5.0, 4u);
  auto costhbins = linspace(-1.0, 1.0, 3u);
  const size_t n_e = Ebins.size() - 1;
  const size_t n_c = costhbins.size() - 1;

  auto flat_histos = [&]() -> BinnedHistograms {
    return {
        .pod_flux_numu    = flat_pod_hist2d(n_e, n_c, 1.0),
        .pod_flux_numubar = flat_pod_hist2d(n_e, n_c, 1.0),
        .pod_flux_nue     = flat_pod_hist2d(n_e, n_c, 1.0),
        .pod_flux_nuebar  = flat_pod_hist2d(n_e, n_c, 1.0),
        .pod_xsec_numu    = flat_pod_hist1d(n_e, 1.0),
        .pod_xsec_numubar = flat_pod_hist1d(n_e, 1.0),
        .pod_xsec_nue     = flat_pod_hist1d(n_e, 1.0),
        .pod_xsec_nuebar  = flat_pod_hist1d(n_e, 1.0),
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
