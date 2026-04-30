#pragma once

#include <TH1D.h>
#include <TH2D.h>
#include <vector>

/// Host-side flux and cross-section histograms, loaded once from
/// data files and passed to whichever model needs them.
struct PhysicsInput {
  TH2D flux_numu, flux_numubar, flux_nue, flux_nuebar;
  TH1D xsec_numu, xsec_numubar, xsec_nue, xsec_nuebar;
  TH1D xsec_nc_nu, xsec_nc_nu_bar;
};

/// Load flux and cross-section data from the global singletons.
/// Defined in physics_input.cxx (host-compiled TU).
PhysicsInput load_physics_input(const std::vector<double> &Ebins,
                                const std::vector<double> &costh_bins,
                                double scale_);
