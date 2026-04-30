#include "ParBinnedInterface.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "fit_config.h"

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>

#include <ROOT/RDataFrame.hxx>
#include <RtypesCore.h>
#include <TRandom.h>
#include <cmath>
#include <print>

int main(int argc, char **argv) {
  TH1::AddDirectory(false);
  std::string outname = argc >= 2 ? argv[1] : "testfit.root";
  auto costheta_bins = linspace(-1., 1., FitConfig::n_costheta_bins + 1);
  auto Ebins = logspace(FitConfig::e_min, FitConfig::e_max, FitConfig::n_energy_bins + 1);

  ParBinnedInterface bint{Ebins, costheta_bins, FitConfig::scale_factor,
                           FitConfig::E_rebin_factor, FitConfig::costh_rebin_factor,
                           FitConfig::ih_bias};
  SimpleDataHist data = bint.GenerateData();

  using vals =
      std::tuple<double, double, double, double, double, double, size_t>;

  auto rawdf = ROOT::RDataFrame{1250000};
  ROOT::RDF::Experimental::AddProgressBar(rawdf);
  size_t count = 0;

  auto df =
      rawdf
          .Define("tuple",
                  [&bint, &data, &count]() -> vals {
                    for (size_t i = 0; i < 5; i++) {
                      auto proposed = bint;
                      proposed.proposeStep();

                      double cur_llh =
                          bint.GetLogLikelihood() +
                          bint.GetLogLikelihoodAgainstData(data);
                      double nxt_llh =
                          proposed.GetLogLikelihood() +
                          proposed.GetLogLikelihoodAgainstData(data);
                      double log_ratio = nxt_llh - cur_llh;

                      if (log_ratio > 0 ||
                          gRandom->Rndm() < std::exp(log_ratio)) {
                        bint = proposed;
                      }
                    }
                    count++;
                    return std::make_tuple(
                        bint.GetDM32sq(), bint.GetDM21sq(),
                        bint.GetT12(), bint.GetT13(), bint.GetT23(),
                        bint.GetDeltaCP(), count);
                  })
          .Define("DM2", [](const vals &t) { return std::get<0>(t); },
                  {"tuple"})
          .Define("Dm2", [](const vals &t) { return std::get<1>(t); },
                  {"tuple"})
          .Define("T12", [](const vals &t) { return std::get<2>(t); },
                  {"tuple"})
          .Define("T13", [](const vals &t) { return std::get<3>(t); },
                  {"tuple"})
          .Define("T23", [](const vals &t) { return std::get<4>(t); },
                  {"tuple"})
          .Define("DCP", [](const vals &t) { return std::get<5>(t); },
                  {"tuple"})
          .Define("count", [](const vals &t) { return std::get<6>(t); },
                  {"tuple"});

  df.Snapshot("tree", outname,
              {"DM2", "Dm2", "T12", "T13", "T23", "DCP", "count"});

  return 0;
}
