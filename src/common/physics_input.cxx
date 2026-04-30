#include "physics_input.h"
#include <hondaflux2d.h>
#include <genie_xsec.h>
#include "constants.h"

extern HondaFlux2D flux_input;
extern genie_xsec xsec_input;

PhysicsInput load_physics_input(const std::vector<double> &Ebins,
                                const std::vector<double> &costh_bins,
                                double scale_) {
  return PhysicsInput{
      .flux_numu =
          flux_input.GetFlux_Hist(Ebins, costh_bins, 14) * scale_,
      .flux_numubar =
          flux_input.GetFlux_Hist(Ebins, costh_bins, -14) * scale_,
      .flux_nue =
          flux_input.GetFlux_Hist(Ebins, costh_bins, 12) * scale_,
      .flux_nuebar =
          flux_input.GetFlux_Hist(Ebins, costh_bins, -12) * scale_,

      .xsec_numu = xsec_input.GetXsecHistMixture(
          Ebins, 14, {{1000060120, 1.0}, {2212, H_to_C}}),
      .xsec_numubar = xsec_input.GetXsecHistMixture(
          Ebins, -14, {{1000060120, 1.0}, {2212, H_to_C}}),
      .xsec_nue = xsec_input.GetXsecHistMixture(
          Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}}),
      .xsec_nuebar = xsec_input.GetXsecHistMixture(
          Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}}),
      .xsec_nc_nu = xsec_input.GetXsecHistMixture(
          Ebins, 12, {{1000060120, 1.0}, {2212, H_to_C}}, false),
      .xsec_nc_nu_bar = xsec_input.GetXsecHistMixture(
          Ebins, -12, {{1000060120, 1.0}, {2212, H_to_C}}, false),
  };
}
