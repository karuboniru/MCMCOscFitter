// Python data export helpers: loads Honda flux and GENIE cross-section data
// as numpy arrays. This TU links against HondaFlux2D + GENIE_XSEC, which are
// intentionally excluded from the main bindings.cxx (BinnedInteractionInject).
//
// Registration into the mcmcoscfitter module happens via init_data_export(),
// called from the main PYBIND11_MODULE block.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constants.h"
#include "genie_xsec.h"
#include "hondaflux2d.h"

#include <TH1.h>
#include <TH2.h>
#include <utility>
#include <vector>

namespace py = pybind11;

// ─── Honda atmospheric flux ──────────────────────────────────────────────────

py::array_t<double>
load_honda_flux_2d(const std::vector<double> &Ebins,
                   const std::vector<double> &costhbins, int pdg,
                   double scale = 1.0) {
  // GetFlux_Hist returns a TH2D. Multiply by scale (default 1.0 means
  // "raw flux" in [m⁻² s⁻¹ sr⁻¹ GeV⁻¹] integrated over bin areas).
  TH2D h = flux_input.GetFlux_Hist(Ebins, costhbins, pdg);
  h.Scale(scale);

  const ssize_t nE = static_cast<ssize_t>(Ebins.size()) - 1;
  const ssize_t nC = static_cast<ssize_t>(costhbins.size()) - 1;
  std::vector<ssize_t> shape = {nE, nC};
  auto arr = py::array_t<double>(shape);
  auto buf = arr.mutable_unchecked<2>();
  for (ssize_t ie = 0; ie < nE; ++ie)
    for (ssize_t ic = 0; ic < nC; ++ic)
      buf(ie, ic) = h.GetBinContent(static_cast<int>(ie + 1),
                                     static_cast<int>(ic + 1));
  return arr;
}

// ─── GENIE cross-section ─────────────────────────────────────────────────────

py::array_t<double>
load_genie_xsec(const std::vector<double> &Ebins, int pdg, bool is_cc = true) {
  // Mixture of C12 (weight 1.0) and H1 (weight H_to_C) for CH₂ scintillator.
  TH1D h =
      xsec_input.GetXsecHistMixture(Ebins, pdg,
                                    {{1000060120, 1.0}, {2212, H_to_C}}, is_cc);

  const ssize_t nE = static_cast<ssize_t>(Ebins.size()) - 1;
  auto arr = py::array_t<double>(nE);
  auto buf = arr.mutable_unchecked<1>();
  for (ssize_t ie = 0; ie < nE; ++ie)
    buf(ie) = h.GetBinContent(static_cast<int>(ie + 1));
  return arr;
}

// ─── PREM Earth density model ────────────────────────────────────────────────
// Returns (radii_km, density_gcm3, Y_e) as three numpy arrays.
// radii are in descending order (outermost first), matching the convention
// used by PREMModel and the GPU kernel.

py::tuple load_prem_density() {
  // From ParProb3ppOscillation constructor:
  //   radii:  {0.0, 1220.0, 3480.0, 5701.0, 6371.0}   — ascending
  //   rhos:   {13.0 * .936, 13.0 * .936, 11.3 * .936, 5.0 * .994, 3.3 * .994}
  //   = density[g/cm³] × 2 × Y_e
  // radius  0 km: inner core, density 13.0 g/cm³, Y_e=0.468, 2·Y_e=0.936
  // radius 1220: inner/outer core boundary
  // radius 3480: core/mantle boundary
  // radius 5701: upper mantle, density 5.0 g/cm³, Y_e=0.497, 2·Y_e=0.994
  // radius 6371: crust/surface, density 3.3 g/cm³, Y_e=0.497, 2·Y_e=0.994

  // Stored in descending order (outermost first) as the PREM model expects.
  std::vector<double> radii    = {6371.0, 5701.0, 3480.0, 1220.0, 0.0};
  std::vector<double> density  = {3.3, 5.0, 11.3, 13.0, 13.0};
  std::vector<double> Y_e      = {0.497, 0.497, 0.468, 0.468, 0.468};

  auto arr_radii = py::array_t<double>(radii.size());
  auto arr_dens  = py::array_t<double>(density.size());
  auto arr_Ye    = py::array_t<double>(Y_e.size());
  std::copy(radii.begin(), radii.end(), arr_radii.mutable_data());
  std::copy(density.begin(), density.end(), arr_dens.mutable_data());
  std::copy(Y_e.begin(), Y_e.end(), arr_Ye.mutable_data());
  return py::make_tuple(arr_radii, arr_dens, arr_Ye);
}

// ─── Convenience: load all physics inputs at once ────────────────────────────
// Returns a dict with keys:
//   flux_numu, flux_numubar, flux_nue, flux_nuebar  (nE, nCos) arrays
//   xsec_numu, xsec_numubar, xsec_nue, xsec_nuebar  (nE,) arrays

py::dict load_physics_input(const std::vector<double> &Ebins,
                            const std::vector<double> &costhbins,
                            double scale = 1.0) {
  py::dict d;
  d["flux_numu"]    = load_honda_flux_2d(Ebins, costhbins, 14, scale);
  d["flux_numubar"] = load_honda_flux_2d(Ebins, costhbins, -14, scale);
  d["flux_nue"]     = load_honda_flux_2d(Ebins, costhbins, 12, scale);
  d["flux_nuebar"]  = load_honda_flux_2d(Ebins, costhbins, -12, scale);
  d["xsec_numu"]    = load_genie_xsec(Ebins, 14);
  d["xsec_numubar"] = load_genie_xsec(Ebins, -14);
  d["xsec_nue"]     = load_genie_xsec(Ebins, 12);
  d["xsec_nuebar"]  = load_genie_xsec(Ebins, -12);
  return d;
}

// ─── Registration helper (called from PYBIND11_MODULE in bindings.cxx) ──────

void init_data_export(py::module_ &m) {
  m.def("load_honda_flux_2d", &load_honda_flux_2d,
        py::arg("Ebins"), py::arg("costhbins"), py::arg("pdg"),
        py::arg("scale") = 1.0,
        "Load Honda atmospheric flux histogram as numpy array (nE, nCos).\n"
        "pdg: 14=numu, -14=numubar, 12=nue, -12=nuebar.\n"
        "scale is applied to the flux values (default 1.0 = raw flux).");

  m.def("load_genie_xsec", &load_genie_xsec,
        py::arg("Ebins"), py::arg("pdg"), py::arg("is_cc") = true,
        "Load GENIE cross-section (CH₂ mixture) as numpy array (nE,).\n"
        "pdg: 14=numu, -14=numubar, 12=nue, -12=nuebar.");

  m.def("load_prem_density", &load_prem_density,
        "Return PREM Earth density model as (radii_km, density_gcm3, Y_e).\n"
        "radii are in descending order (outermost first).");

  m.def("load_physics_input", &load_physics_input,
        py::arg("Ebins"), py::arg("costhbins"), py::arg("scale") = 1.0,
        "Load all physics inputs (flux + xsec) as a dict of numpy arrays.\n"
        "Returns keys: flux_numu, flux_numubar, flux_nue, flux_nuebar, "
        "xsec_numu, xsec_numubar, xsec_nue, xsec_nuebar.");
}
