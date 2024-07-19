#include "SimpleInteraction.h"
#include "OscillationParameters.h"
#include "genie_xsec.h"
#include "hondaflux.h"
#include <TF3.h>

HondaFlux SimpleInteraction::flux_input("/var/home/yan/neutrino/honda-3d.txt");
genie_xsec SimpleInteraction::xsec_input(
    "/var/home/yan/neutrino/spline/full/genie_spline_3_0_2/G18_02b_02_11b/"
    "3.04.02-routine_validation_01-xsec_vA/total_xsec.root");

double
SimpleInteraction::GetLogLikelihoodAgainstData(const StateI &dataset) const {
  double llh = 0;
  for (const auto &data_point :
       dynamic_cast<const DataSet<SimpleDataPoint> &>(dataset)) {
    auto prob = GetProb(data_point.flavor, data_point.E, data_point.costheta);
    int symbol = data_point.flavor > 0 ? 1 : -1;
    double flux_numu = flux_input.GetFlux(data_point.E, data_point.costheta,
                                          data_point.phi, symbol * 14);
    double flux_nue = flux_input.GetFlux(data_point.E, data_point.costheta,
                                         data_point.phi, symbol * 12);
    double xsec =
        xsec_input.GetXsec(data_point.E, data_point.flavor, 1000060120);
    size_t flux_target = abs(data_point.flavor) == 14 ? 1 : 0;
    llh += log(flux_numu * prob[1][flux_target] +
               flux_nue * prob[0][flux_target]) +
           log(xsec);
  }
  return llh - weight_int;
}

void SimpleInteraction::proposeStep() {
  OscillationParameters::proposeStep();
  auto weight = [&](double E, double costheta, double phi, int flavor) {
    int symbol = flavor > 0 ? 1 : -1;
    double flux_numu = flux_input.GetFlux(E, costheta, phi, symbol * 14);
    double flux_nue = flux_input.GetFlux(E, costheta, phi, symbol * 12);
    double xsec = xsec_input.GetXsec(E, flavor, 1000060120);

    auto prob = GetProb(flavor, E, costheta);
    size_t flux_target = abs(flavor) == 14 ? 1 : 0;
    return (flux_numu * prob[1][flux_target] +
            flux_nue * prob[0][flux_target]) *
           xsec;
  };
  constexpr double Emin = 3, Emax = 10;
  TF3 weight_func(
      "weight_func",
      [&weight](double *x, double *) {
        double E = x[0];
        double costh = x[1];
        double phi = x[2];
        return weight(E, costh, phi, 14) + weight(E, costh, phi, -14) +
               weight(E, costh, phi, -12) + weight(E, costh, phi, 12);
      },
      Emin, Emax, -1, 1, 0, 2 * TMath::Pi(), 0);
  // weight_func.SetNpx(80);
  // weight_func.SetNpy(300);
  weight_func.SetNpz(12);
  weight_int =
      weight_func.Integral(Emin, Emax, -1, 1, 0, 2 * TMath::Pi(), 0.1) / 100;
  std::cerr << "weight_int: " << weight_int << std::endl;
}