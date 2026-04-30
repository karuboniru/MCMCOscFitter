#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BinnedInteraction.h"
#include "IHistogramPropagator.h"
#include "ModelDataLLH.h"
#include "OscillationParameters.h"
#include "ParProb3ppOscillation.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "chi2.h"
#include "constants.h"
#include "walker.h"

#include <TH1.h>
#include <TH2.h>
#include <TRandom.h>
#include <cmath>
#include <memory>

namespace py = pybind11;

// ─── numpy ↔ POD helpers (Phase 6: direct, no TH2D intermediate) ────────────

static PodHist2D<oscillaton_calc_precision>
array_to_pod_hist2d(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                    size_t n_costh, size_t n_e) {
  auto buf = arr.unchecked<2>();
  PodHist2D<oscillaton_calc_precision> pod(n_costh, n_e);
  for (size_t c = 0; c < n_costh; ++c)
    for (size_t e = 0; e < n_e; ++e)
      pod(c, e) = static_cast<oscillaton_calc_precision>(buf(static_cast<ssize_t>(e), static_cast<ssize_t>(c)));
  return pod;
}

static PodHist1D
array_to_pod_hist1d(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
  auto buf = arr.unchecked<1>();
  PodHist1D pod(static_cast<size_t>(buf.shape(0)));
  for (ssize_t i = 0; i < buf.shape(0); ++i)
    pod[static_cast<size_t>(i)] = static_cast<oscillaton_calc_precision>(buf(i));
  return pod;
}

static py::array_t<double> pod_hist2d_to_array(const PodHist2D<double> &pod) {
  std::vector<ssize_t> shape = {static_cast<ssize_t>(pod.n_e), static_cast<ssize_t>(pod.n_costh)};
  auto result = py::array_t<double>(shape);
  auto buf = result.template mutable_unchecked<2>();
  for (size_t c = 0; c < pod.n_costh; ++c)
    for (size_t e = 0; e < pod.n_e; ++e)
      buf(static_cast<ssize_t>(e), static_cast<ssize_t>(c)) = pod(c, e);
  return result;
}

// ─── numpy ↔ ROOT helpers (legacy, kept for backward compat) ─────────────────

static TH2D array_to_th2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr,
    const std::vector<double> &Ebins, const std::vector<double> &costhbins,
    const char *name = "h") {
  auto buf = arr.unchecked<2>();
  TH2D h(name, "", static_cast<int>(Ebins.size()) - 1, Ebins.data(),
         static_cast<int>(costhbins.size()) - 1, costhbins.data());
  for (int x = 0; x < buf.shape(0); ++x)
    for (int y = 0; y < buf.shape(1); ++y)
      h.SetBinContent(x + 1, y + 1, buf(x, y));
  return h;
}

static py::array_t<double> th2d_to_array(const TH2D &h) {
  const int nx = h.GetNbinsX(), ny = h.GetNbinsY();
  auto result = py::array_t<double>({nx, ny});
  auto buf = result.mutable_unchecked<2>();
  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y)
      buf(x, y) = h.GetBinContent(x + 1, y + 1);
  return result;
}

static TH1D array_to_th1d(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr,
    const std::vector<double> &Ebins, const char *name = "h1") {
  auto buf = arr.unchecked<1>();
  TH1D h(name, "", static_cast<int>(Ebins.size()) - 1, Ebins.data());
  for (int x = 0; x < buf.shape(0); ++x)
    h.SetBinContent(x + 1, buf(x));
  return h;
}

// ─── Propagator trampoline ───────────────────────────────────────────────────
// Python subclasses override get_prob_hists / get_prob_hists_3f which receive
// and return numpy arrays, hiding the TH2D implementation detail.

class PyPropagator : public IHistogramPropagator {
public:
  void re_calculate(const OscillationParameters &) override {
    // Default no-op: Python custom propagators compute on demand in get_prob_hists.
    py::gil_scoped_acquire gil;
    py::function fn = py::get_override(this, "re_calculate");
    if (fn) fn();
  }

  // Python signature:
  //   get_prob_hists(Ebins, costhbins, params) -> np.ndarray shape (2,2,2,nE,nCosth)
  //   axis 0: nu/antinu   axis 1: from (nue=0, numu=1)   axis 2: to
  std::array<std::array<std::array<TH2D, 2>, 2>, 2>
  GetProb_Hists(const std::vector<double> &Ebins,
                const std::vector<double> &costhbins,
                const OscillationParameters &p) override {
    py::gil_scoped_acquire gil;
    py::function fn = py::get_override(this, "get_prob_hists");
    if (!fn)
      throw std::runtime_error(
          "PyPropagator: get_prob_hists() not implemented");

    auto raw = fn(Ebins, costhbins, py::cast(p, py::return_value_policy::reference))
                   .cast<py::array_t<double,
                                     py::array::c_style | py::array::forcecast>>();
    auto buf = raw.unchecked<5>(); // (2, 2, 2, nE, nCosth)
    const int nE = static_cast<int>(Ebins.size()) - 1;
    const int nC = static_cast<int>(costhbins.size()) - 1;
    std::array<std::array<std::array<TH2D, 2>, 2>, 2> ret;
    for (int nu = 0; nu < 2; ++nu)
      for (int f = 0; f < 2; ++f)
        for (int t = 0; t < 2; ++t) {
          TH2D &h = ret[nu][f][t];
          h = TH2D("", "", nE, Ebins.data(), nC, costhbins.data());
          for (int e = 0; e < nE; ++e)
            for (int c = 0; c < nC; ++c)
              h.SetBinContent(e + 1, c + 1, buf(nu, f, t, e, c));
        }
    return ret;
  }

  // Python signature:
  //   get_prob_hists_3f(Ebins, costhbins, params) -> np.ndarray shape (2,3,3,nE,nCosth)
  std::array<std::array<std::array<TH2D, 3>, 3>, 2>
  GetProb_Hists_3F(const std::vector<double> &Ebins,
                   const std::vector<double> &costhbins,
                   const OscillationParameters &p) override {
    py::gil_scoped_acquire gil;
    py::function fn = py::get_override(this, "get_prob_hists_3f");
    if (!fn)
      throw std::runtime_error(
          "PyPropagator: get_prob_hists_3f() not implemented");

    auto raw = fn(Ebins, costhbins, py::cast(p, py::return_value_policy::reference))
                   .cast<py::array_t<double,
                                     py::array::c_style | py::array::forcecast>>();
    auto buf = raw.unchecked<5>(); // (2, 3, 3, nE, nCosth)
    const int nE = static_cast<int>(Ebins.size()) - 1;
    const int nC = static_cast<int>(costhbins.size()) - 1;
    std::array<std::array<std::array<TH2D, 3>, 3>, 2> ret;
    for (int nu = 0; nu < 2; ++nu)
      for (int f = 0; f < 3; ++f)
        for (int t = 0; t < 3; ++t) {
          TH2D &h = ret[nu][f][t];
          h = TH2D("", "", nE, Ebins.data(), nC, costhbins.data());
          for (int e = 0; e < nE; ++e)
            for (int c = 0; c < nC; ++c)
              h.SetBinContent(e + 1, c + 1, buf(nu, f, t, e, c));
        }
    return ret;
  }
};

// ─── Module ──────────────────────────────────────────────────────────────────

PYBIND11_MODULE(mcmcoscfitter, m) {
  m.doc() = R"doc(
    MCMCOscFitter Python bindings.

    Quick-start example (injectable constructor, no physics data files needed)::

        import mcmcoscfitter as mof
        import numpy as np

        Ebins    = mof.logspace(0.1, 20.0, 51)   # 50 E-bins
        costhbins = mof.linspace(-1.0, 1.0, 21)  # 20 costh-bins
        centers_E = mof.to_center(Ebins)

        prop = mof.ParProb3ppOscillation(centers_E,
                                         mof.to_center(costhbins))

        flux   = np.ones((50, 20))  # placeholder flux [E x costh]
        xsec   = np.ones(50)        # placeholder xsec [E]

        histos = mof.BinnedHistograms(
            flux_numu=flux, flux_numubar=flux, flux_nue=flux, flux_nuebar=flux,
            xsec_numu=xsec, xsec_numubar=xsec, xsec_nue=xsec, xsec_nuebar=xsec,
            Ebins=Ebins, costhbins=costhbins)

        model = mof.BinnedInteraction(Ebins, costhbins, prop, histos)
        data  = model.generate_data()

        import copy
        current = model
        for _ in range(10000):
            nxt = copy.deepcopy(current)
            nxt.propose_step()
            if mof.mcmc_accept(current, nxt, data):
                current = nxt

  )doc";

  // ── Constants ──────────────────────────────────────────────────────────────
  m.attr("scale_factor_1y") = scale_factor_1y;
  m.attr("scale_factor_6y") = scale_factor_6y;
  m.attr("H_to_C") = H_to_C;
  m.attr("atmo_count_C12") = atmo_count_C12;

  // ── Binning utilities ──────────────────────────────────────────────────────
  m.def("linspace", &linspace<double>, py::arg("Emin"), py::arg("Emax"),
        py::arg("n"),
        "Return n evenly-spaced values from Emin to Emax (inclusive).");
  m.def("logspace", &logspace<double>, py::arg("Emin"), py::arg("Emax"),
        py::arg("n"),
        "Return n logarithmically-spaced values from Emin to Emax.");
  m.def("to_center",
        [](const std::vector<double> &v) { return to_center<double>(v); },
        py::arg("edges"),
        "Return arithmetic midpoints of adjacent bin edges.");
  m.def("to_center_g",
        [](const std::vector<double> &v) { return to_center_g<double>(v); },
        py::arg("edges"),
        "Return geometric midpoints of adjacent bin edges.");
  m.def("divide_bins",
        [](const std::vector<double> &v, size_t n) {
          return divide_bins<double>(v, n);
        },
        py::arg("edges"), py::arg("multiplier"),
        "Subdivide each bin into *multiplier* equal sub-bins (linear).");
  m.def("divide_bins_log",
        [](const std::vector<double> &v, size_t n) {
          return divide_bins_log<double>(v, n);
        },
        py::arg("edges"), py::arg("multiplier"),
        "Subdivide each bin into *multiplier* equal sub-bins (log-spaced).");

  // ── chi2 statistic ─────────────────────────────────────────────────────────
  m.def(
      "poisson_chi2",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> data,
         py::array_t<double, py::array::c_style | py::array::forcecast> pred,
         const std::vector<double> &Ebins,
         const std::vector<double> &costhbins) {
        TH1::AddDirectory(false);
        return TH2D_chi2(array_to_th2d(data, Ebins, costhbins, "d"),
                         array_to_th2d(pred, Ebins, costhbins, "p"));
      },
      py::arg("data"), py::arg("pred"), py::arg("Ebins"), py::arg("costhbins"),
      R"doc(Poisson log-likelihood chi2: 2*sum(pred - data + data*ln(data/pred)).
Returns 0 when data == pred everywhere.)doc");

  // ── param struct ───────────────────────────────────────────────────────────
  py::class_<param>(m, "Param")
      .def(py::init<>())
      .def(py::init([](double DM2, double Dm2, double T23, double T13,
                       double T12, double DCP) {
             return param{DM2, Dm2, T23, T13, T12, DCP};
           }),
           py::arg("DM2"), py::arg("Dm2"), py::arg("T23"), py::arg("T13"),
           py::arg("T12"), py::arg("DCP"))
      .def_readwrite("DM2", &param::DM2)
      .def_readwrite("Dm2", &param::Dm2)
      .def_readwrite("T23", &param::T23)
      .def_readwrite("T13", &param::T13)
      .def_readwrite("T12", &param::T12)
      .def_readwrite("DCP", &param::DCP)
      .def("__repr__", [](const param &p) {
        return "<Param DM2=" + std::to_string(p.DM2) +
               " Dm2=" + std::to_string(p.Dm2) +
               " T23=" + std::to_string(p.T23) +
               " T13=" + std::to_string(p.T13) +
               " T12=" + std::to_string(p.T12) +
               " DCP=" + std::to_string(p.DCP) + ">";
      });

  // ── PullToggle ─────────────────────────────────────────────────────────────
  py::class_<pull_toggle>(m, "PullToggle",
                          "Boolean mask selecting which oscillation-parameter "
                          "priors are active. Indices: 0=DM32, 1=DM21, 2=T23, "
                          "3=T13, 4=T12, 5=DCP.")
      .def(py::init<>())
      .def("__getitem__",
           [](const pull_toggle &t, size_t i) { return t[i]; })
      .def("__setitem__",
           [](pull_toggle &t, size_t i, bool v) { t[i] = v; })
      .def("active", &pull_toggle::get_active,
           "Names of currently-active priors.")
      .def("inactive", &pull_toggle::get_inactive,
           "Names of currently-inactive priors.")
      .def_readonly_static("names", &pull_toggle::names);

  m.attr("all_on")      = all_on;
  m.attr("all_off")     = all_off;
  m.attr("SK_w_T13")    = SK_w_T13;
  m.attr("SK_wo_T13")   = SK_wo_T13;

  // ── StateI (abstract — only exposed so pybind11 knows the hierarchy) ───────
  py::class_<StateI>(m, "StateI");

  // ── OscillationParameters ──────────────────────────────────────────────────
  py::class_<OscillationParameters>(m, "OscillationParameters",
      "PMNS oscillation parameters with Gaussian priors (PDG 2023 values).")
      .def(py::init<>())
      .def("propose_step", py::overload_cast<>(&OscillationParameters::proposeStep),
           "Randomly perturb parameters (Gaussian step). "
           "Hierarchy can flip with probability 0.2.")
      .def("log_likelihood",
           py::overload_cast<>(&OscillationParameters::GetLogLikelihood,
                               py::const_),
           "Gaussian prior log-likelihood using the active PullToggle.")
      .def("log_likelihood",
           py::overload_cast<const pull_toggle &>(
               &OscillationParameters::GetLogLikelihood, py::const_),
           py::arg("toggle"),
           "Gaussian prior log-likelihood with an explicit PullToggle.")
      .def("flip_hierarchy", &OscillationParameters::flip_hierarchy,
           "Switch between normal (DM32>0) and inverted (DM32<0) hierarchy.")
      .def("set_param", &OscillationParameters::set_param, py::arg("p"),
           "Set all six parameters at once. Sign of DM2 determines hierarchy.")
      .def("set_toggle", &OscillationParameters::set_toggle, py::arg("toggle"))
      .def("get_toggle", &OscillationParameters::get_toggle,
           py::return_value_policy::copy)
      .def("set_proposal_distance", &OscillationParameters::set_proposal_distance,
           py::arg("d"), "Set MCMC proposal step size factor (default 0.1).")
      .def("get_proposal_distance", &OscillationParameters::get_proposal_distance,
           "Get current proposal step size factor.")
      .def_property_readonly("DM32sq", &OscillationParameters::GetDM32sq,
                             "Δm²₃₂ (>0 NH, <0 IH) in eV².")
      .def_property_readonly("DM21sq", &OscillationParameters::GetDM21sq,
                             "Δm²₂₁ in eV².")
      .def_property_readonly("T23", &OscillationParameters::GetT23,
                             "sin²θ₂₃.")
      .def_property_readonly("T13", &OscillationParameters::GetT13,
                             "sin²θ₁₃.")
      .def_property_readonly("T12", &OscillationParameters::GetT12,
                             "sin²θ₁₂.")
      .def_property_readonly("DeltaCP", &OscillationParameters::GetDeltaCP,
                             "δ_CP in radians.")
      .def_property_readonly("is_NH",
                             [](const OscillationParameters &p) {
                               return p.GetDM32sq() > 0;
                             },
                             "True if normal hierarchy.")
      .def("__repr__", [](const OscillationParameters &p) {
        return "<OscillationParameters DM32sq=" +
               std::to_string(p.GetDM32sq()) +
               " T23=" + std::to_string(p.GetT23()) +
               " T13=" + std::to_string(p.GetT13()) +
               " DeltaCP=" + std::to_string(p.GetDeltaCP()) + ">";
      });

  // ── PropagatorBase (Python-subclassable IHistogramPropagator) ─────────────
  py::class_<IHistogramPropagator, PyPropagator,
             std::shared_ptr<IHistogramPropagator>>(
      m, "PropagatorBase",
      R"doc(Abstract base for oscillation probability propagators.

Override get_prob_hists and get_prob_hists_3f to implement a custom propagator
usable with BinnedInteraction's injectable constructor.

get_prob_hists(Ebins, costhbins, params) -> np.ndarray shape (2, 2, 2, nE, nCosth)
  axis 0: neutrino (0) / antineutrino (1)
  axis 1: from-flavor  nue=0, numu=1
  axis 2: to-flavor    nue=0, numu=1

get_prob_hists_3f(Ebins, costhbins, params) -> np.ndarray shape (2, 3, 3, nE, nCosth)
  same but 3 flavours (nue=0, numu=1, nutau=2)
)doc")
      .def(py::init<>())
      .def("get_prob_hists", [](IHistogramPropagator &) {},
           "Override in Python subclass.")
      .def("get_prob_hists_3f", [](IHistogramPropagator &) {},
           "Override in Python subclass.")
      .def("re_calculate", [](IHistogramPropagator &prop, const OscillationParameters &p) {
             prop.re_calculate(p);
           }, py::arg("params"),
           "Pre-compute oscillation probabilities. Default no-op; override if needed.");

  // ── ParProb3ppOscillation (concrete propagator) ───────────────────────────
  py::class_<ParProb3ppOscillation, IHistogramPropagator,
             std::shared_ptr<ParProb3ppOscillation>>(
      m, "ParProb3ppOscillation",
      "Parallelised (OpenMP / CUDA) PMNS propagator through an Earth density "
      "profile. Pass bin centres (not edges) to the constructor.")
      .def(py::init([](const std::vector<double> &Ecenters,
                       const std::vector<double> &costhcenters) {
             std::vector<float> Ef(Ecenters.begin(), Ecenters.end());
             std::vector<float> Cf(costhcenters.begin(), costhcenters.end());
             return std::make_shared<ParProb3ppOscillation>(Ef, Cf);
           }),
           py::arg("E_centers"), py::arg("costh_centers"))
      .def(
          "get_prob_hists",
          [](ParProb3ppOscillation &prop, const std::vector<double> &Ebins,
             const std::vector<double> &costhbins,
             const OscillationParameters &p) {
            // Uses POD path — avoids TH2D construction.
            auto hists = prop.GetProb_Hists_POD(Ebins, costhbins, p);
            const ssize_t nE = static_cast<ssize_t>(Ebins.size()) - 1;
            const ssize_t nC = static_cast<ssize_t>(costhbins.size()) - 1;
            std::vector<ssize_t> shape = {2, 2, 2, nE, nC};
            auto arr = py::array_t<double>(shape);
            auto buf = arr.template mutable_unchecked<5>();
            for (int nu = 0; nu < 2; ++nu)
              for (int f = 0; f < 2; ++f)
                for (int t = 0; t < 2; ++t)
                  for (ssize_t c = 0; c < nC; ++c)
                    for (ssize_t e = 0; e < nE; ++e)
                      buf(nu, f, t, e, c) = hists[nu][f][t](static_cast<size_t>(c), static_cast<size_t>(e));
            return arr;
          },
          py::arg("Ebins"), py::arg("costhbins"), py::arg("params"),
          "Compute oscillation probabilities, shape (2, 2, 2, nE, nCosth).")
      .def(
          "get_prob_hists_3f",
          [](ParProb3ppOscillation &prop, const std::vector<double> &Ebins,
             const std::vector<double> &costhbins,
             const OscillationParameters &p) {
            auto hists = prop.GetProb_Hists_3F_POD(Ebins, costhbins, p);
            const ssize_t nE = static_cast<ssize_t>(Ebins.size()) - 1;
            const ssize_t nC = static_cast<ssize_t>(costhbins.size()) - 1;
            std::vector<ssize_t> shape = {2, 3, 3, nE, nC};
            auto arr = py::array_t<double>(shape);
            auto buf = arr.template mutable_unchecked<5>();
            for (int nu = 0; nu < 2; ++nu)
              for (int f = 0; f < 3; ++f)
                for (int t = 0; t < 3; ++t)
                  for (ssize_t c = 0; c < nC; ++c)
                    for (ssize_t e = 0; e < nE; ++e)
                      buf(nu, f, t, e, c) = hists[nu][f][t](static_cast<size_t>(c), static_cast<size_t>(e));
            return arr;
          },
          py::arg("Ebins"), py::arg("costhbins"), py::arg("params"),
          "3-flavour probabilities, shape (2, 3, 3, nE, nCosth).");

  // ── BinnedHistograms helper struct ─────────────────────────────────────────
  // Python-side constructor converts numpy arrays to POD directly (avoids
  // numpy→TH2D→POD double conversion).  Also fills TH2D for backward compat.
  py::class_<BinnedHistograms>(m, "BinnedHistograms",
      "Container for pre-computed flux (TH2D) and cross-section (TH1D) "
      "histograms.  Construct from numpy arrays via the keyword-argument "
      "factory below; Ebins and costhbins define the bin edges.")
      .def(py::init([](py::array_t<double> flux_numu,
                       py::array_t<double> flux_numubar,
                       py::array_t<double> flux_nue,
                       py::array_t<double> flux_nuebar,
                       py::array_t<double> xsec_numu,
                       py::array_t<double> xsec_numubar,
                       py::array_t<double> xsec_nue,
                       py::array_t<double> xsec_nuebar,
                       const std::vector<double> &Ebins,
                       const std::vector<double> &costhbins) {
             TH1::AddDirectory(false);
             const size_t n_e = Ebins.size() - 1;
             const size_t n_c = costhbins.size() - 1;
             BinnedHistograms h;
             // Fill POD directly from numpy (fast path for computation).
             h.pod_flux_numu    = array_to_pod_hist2d(flux_numu,    n_c, n_e);
             h.pod_flux_numubar = array_to_pod_hist2d(flux_numubar, n_c, n_e);
             h.pod_flux_nue     = array_to_pod_hist2d(flux_nue,     n_c, n_e);
             h.pod_flux_nuebar  = array_to_pod_hist2d(flux_nuebar,  n_c, n_e);
             h.pod_xsec_numu    = array_to_pod_hist1d(xsec_numu);
             h.pod_xsec_numubar = array_to_pod_hist1d(xsec_numubar);
             h.pod_xsec_nue     = array_to_pod_hist1d(xsec_nue);
             h.pod_xsec_nuebar  = array_to_pod_hist1d(xsec_nuebar);
             h.pod_valid = true;
             // Also fill TH2D/TH1D for backward compat (I/O, plotting, etc.).
             h.flux_numu    = array_to_th2d(flux_numu,    Ebins, costhbins, "fn");
             h.flux_numubar = array_to_th2d(flux_numubar, Ebins, costhbins, "fnb");
             h.flux_nue     = array_to_th2d(flux_nue,     Ebins, costhbins, "fe");
             h.flux_nuebar  = array_to_th2d(flux_nuebar,  Ebins, costhbins, "feb");
             h.xsec_numu    = array_to_th1d(xsec_numu,    Ebins, "xn");
             h.xsec_numubar = array_to_th1d(xsec_numubar, Ebins, "xnb");
             h.xsec_nue     = array_to_th1d(xsec_nue,     Ebins, "xe");
             h.xsec_nuebar  = array_to_th1d(xsec_nuebar,  Ebins, "xeb");
             return h;
           }),
           py::arg("flux_numu"), py::arg("flux_numubar"),
           py::arg("flux_nue"),  py::arg("flux_nuebar"),
           py::arg("xsec_numu"), py::arg("xsec_numubar"),
           py::arg("xsec_nue"),  py::arg("xsec_nuebar"),
           py::arg("Ebins"),     py::arg("costhbins"));

  // ── DataHist ───────────────────────────────────────────────────────────────
  py::class_<SimpleDataHist>(m, "DataHist",
      "Four 2-D event-count histograms (numu, numubar, nue, nuebar).  "
      "Bin contents are accessible as numpy arrays.")
      .def(py::init<>())
      .def_property(
          "numu",
          [](const SimpleDataHist &d) { return th2d_to_array(d.hist_numu); },
          [](SimpleDataHist &d,
             py::array_t<double, py::array::c_style | py::array::forcecast> a) {
            const int nx = d.hist_numu.GetNbinsX();
            const int ny = d.hist_numu.GetNbinsY();
            auto buf = a.unchecked<2>();
            for (int x = 0; x < nx; ++x)
              for (int y = 0; y < ny; ++y)
                d.hist_numu.SetBinContent(x + 1, y + 1, buf(x, y));
          })
      .def_property(
          "numubar",
          [](const SimpleDataHist &d) {
            return th2d_to_array(d.hist_numubar);
          },
          [](SimpleDataHist &d,
             py::array_t<double, py::array::c_style | py::array::forcecast> a) {
            const int nx = d.hist_numubar.GetNbinsX();
            const int ny = d.hist_numubar.GetNbinsY();
            auto buf = a.unchecked<2>();
            for (int x = 0; x < nx; ++x)
              for (int y = 0; y < ny; ++y)
                d.hist_numubar.SetBinContent(x + 1, y + 1, buf(x, y));
          })
      .def_property(
          "nue",
          [](const SimpleDataHist &d) { return th2d_to_array(d.hist_nue); },
          [](SimpleDataHist &d,
             py::array_t<double, py::array::c_style | py::array::forcecast> a) {
            const int nx = d.hist_nue.GetNbinsX();
            const int ny = d.hist_nue.GetNbinsY();
            auto buf = a.unchecked<2>();
            for (int x = 0; x < nx; ++x)
              for (int y = 0; y < ny; ++y)
                d.hist_nue.SetBinContent(x + 1, y + 1, buf(x, y));
          })
      .def_property(
          "nuebar",
          [](const SimpleDataHist &d) { return th2d_to_array(d.hist_nuebar); },
          [](SimpleDataHist &d,
             py::array_t<double, py::array::c_style | py::array::forcecast> a) {
            const int nx = d.hist_nuebar.GetNbinsX();
            const int ny = d.hist_nuebar.GetNbinsY();
            auto buf = a.unchecked<2>();
            for (int x = 0; x < nx; ++x)
              for (int y = 0; y < ny; ++y)
                d.hist_nuebar.SetBinContent(x + 1, y + 1, buf(x, y));
          })
      .def("save", [](const SimpleDataHist &d, const std::string &f) {
        d.SaveAs(f.c_str());
      }, py::arg("filename"))
      .def("load", [](SimpleDataHist &d, const std::string &f) {
        d.LoadFrom(f.c_str());
      }, py::arg("filename"))
      .def("round", &SimpleDataHist::Round,
           "Round all bin contents to the nearest integer (Poisson sampling).");

  // ── BinnedInteraction ──────────────────────────────────────────────────────
  py::class_<BinnedInteraction, OscillationParameters>(m, "BinnedInteraction",
      R"doc(Binned likelihood model combining oscillation probabilities,
atmospheric flux, and neutrino cross-sections.

Two constructors are available:

  BinnedInteraction(Ebins, costheta_bins, propagator, histos,
                    E_rebin=1, costh_rebin=1, IH_bias=1.0)
      Injectable constructor — no physics data files required at import time.
      Pass a ParProb3ppOscillation (or PropagatorBase subclass) and a
      BinnedHistograms object built from numpy arrays.

The production constructor (requires HondaFlux2D + GENIE_XSEC data files)
is available from the C++ side only; use the C++ executables or the full
BinnedInteraction CMake target for that path.
)doc")
      // Injectable constructor
      .def(py::init([](const std::vector<double> &Ebins,
                       const std::vector<double> &costhbins,
                       std::shared_ptr<IHistogramPropagator> prop,
                       BinnedHistograms histos,
                       size_t E_rebin, size_t costh_rebin,
                       double IH_bias) {
             TH1::AddDirectory(false);
             return BinnedInteraction(Ebins, costhbins, std::move(prop),
                                     std::move(histos), E_rebin, costh_rebin,
                                     IH_bias);
           }),
           py::arg("Ebins"), py::arg("costheta_bins"), py::arg("propagator"),
           py::arg("histos"), py::arg("E_rebin") = 1,
           py::arg("costh_rebin") = 1, py::arg("IH_bias") = 1.0)
      .def("propose_step", &BinnedInteraction::proposeStep,
           "Perturb oscillation parameters and recompute prediction.")
      .def("update_prediction", &BinnedInteraction::UpdatePrediction,
           "Recompute event-rate prediction from current parameters.")
      .def("log_likelihood", &BinnedInteraction::GetLogLikelihood,
           "Oscillation-parameter prior log-likelihood (+ IH bias if IH).")
      .def("log_likelihood_against_data",
           &BinnedInteraction::GetLogLikelihoodAgainstData,
           py::arg("data"),
           "-0.5 * Poisson chi2 summed over all four flavour channels.")
      .def("generate_data", &BinnedInteraction::GenerateData,
           "Return an Asimov DataHist matching the current prediction.")
      .def("generate_data_noosc", &BinnedInteraction::GenerateData_NoOsc,
           "Return a DataHist with no oscillation applied.")
      .def("flip_hierarchy", &BinnedInteraction::flip_hierarchy,
           "Switch NH↔IH and recompute the prediction.")
      .def("save", [](const BinnedInteraction &b, const std::string &f) {
        b.SaveAs(f.c_str());
      }, py::arg("filename"), "Write flux, xsec, and prediction histograms to a ROOT file.")
      .def_property_readonly("DM32sq", &BinnedInteraction::GetDM32sq)
      .def_property_readonly("DM21sq", &BinnedInteraction::GetDM21sq)
      .def_property_readonly("T23",    &BinnedInteraction::GetT23)
      .def_property_readonly("T13",    &BinnedInteraction::GetT13)
      .def_property_readonly("T12",    &BinnedInteraction::GetT12)
      .def_property_readonly("DeltaCP",&BinnedInteraction::GetDeltaCP)
      .def_property_readonly("is_NH",
                             [](const BinnedInteraction &b) {
                               return b.GetDM32sq() > 0;
                             })
      // copy support for the MCMC pattern: next = copy.deepcopy(current)
      .def("__copy__",
           [](const BinnedInteraction &self) { return BinnedInteraction(self); })
      .def("__deepcopy__", [](const BinnedInteraction &self, py::dict) {
        return BinnedInteraction(self);
      });

  // ── MCMC acceptance ────────────────────────────────────────────────────────
  m.def(
      "mcmc_accept",
      [](const BinnedInteraction &current, const BinnedInteraction &nxt,
         const SimpleDataHist &data) {
        double cur_llh = current.GetLogLikelihood() +
                         current.GetLogLikelihoodAgainstData(data);
        double nxt_llh =
            nxt.GetLogLikelihood() + nxt.GetLogLikelihoodAgainstData(data);
        double log_ratio = nxt_llh - cur_llh;
        if (log_ratio > 0)
          return true;
        return gRandom->Rndm() < std::exp(log_ratio);
      },
      py::arg("current"), py::arg("proposed"), py::arg("data"),
      R"doc(Metropolis-Hastings acceptance test.

Computes total log-likelihood = model.log_likelihood() +
model.log_likelihood_against_data(data) for both states and accepts the
proposed state with probability min(1, exp(llh_proposed - llh_current)).
)doc");

  // Generic version operating on any StateI (e.g. OscillationParameters alone).
  m.def("mcmc_accept_state",
        [](const StateI &current, const StateI &proposed) {
          return MCMCAcceptState(current, proposed);
        },
        py::arg("current"), py::arg("proposed"),
        "Generic Metropolis acceptance using StateI::GetLogLikelihood() only.");

  // ── Random seed ────────────────────────────────────────────────────────────
  m.def("set_seed", [](unsigned int seed) { gRandom->SetSeed(seed); },
        py::arg("seed"), "Set the ROOT random seed used by propose_step and mcmc_accept.");
}
