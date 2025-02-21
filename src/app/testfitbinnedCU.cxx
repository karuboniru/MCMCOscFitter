#include "ParBinnedInterface.h"
#include "SimpleDataHist.h"
#include "binning_tool.hpp"
#include "constants.h"
#include "timer.hpp"
#include "walker.h"

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include <RtypesCore.h>
#include <TMath.h>
#include <TRandom.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <print>

int main(int argc, char **argv) {
  //   ROOT::EnableImplicitMT(10);
  TH1::AddDirectory(false);
  std::string outname = argc == 2 ? argv[1] : "testfit.root";
  auto costheta_bins = linspace(-1., 1., 481);

  auto Ebins = logspace(0.1, 20., 401);

  constexpr double scale_factor =
      (2e10 / (12 + H_to_C) * 6.02214076e23) * // number of target C12
      ((6 * 365) * 24 * 3600) /                // seconds in a year
      1e42; // unit conversion from 1e-38 cm^2 to 1e-42 m^2

  ParBinnedInterface bint{Ebins, costheta_bins, scale_factor, 40, 40, 8000};
  auto cdata = bint.GenerateData();
  //   cdata.Round();

  using combined_type = ModelAndData<ParBinnedInterface, SimpleDataHist>;
  using vals =
      std::tuple<double, double, double, double, double, double, size_t>;
  std::vector<combined_type> state_pool{};
  auto nth = ROOT::GetThreadPoolSize() == 0 ? 1 : ROOT::GetThreadPoolSize();
  for (size_t i = 0; i < nth; i++) {
    state_pool.emplace_back(bint, cdata).proposeStep();
  }
  auto rawdf = ROOT::RDataFrame{235000};
  ROOT::RDF::Experimental::AddProgressBar(rawdf);
  std::atomic<size_t> count{};
  auto df =
      rawdf
          .Define("tuple",
                  [&state_pool, &count](unsigned int id) -> vals {
                    auto &current_state = state_pool[id];
                    //     TimeCount timer{"3 step"};
                    for (size_t i = 0; i < 3; i++) {
                      auto new_state = current_state;
                      new_state.proposeStep();
                      if (MCMCAcceptState(current_state, new_state)) {
                        current_state = new_state;
                      }
                    }
                    count++;
                    return std::make_tuple(
                        current_state.GetModel().GetDM32sq(),
                        current_state.GetModel().GetDM21sq(),
                        current_state.GetModel().GetT12(),
                        current_state.GetModel().GetT13(),
                        current_state.GetModel().GetT23(),
                        current_state.GetModel().GetDeltaCP(), count.load());
                  },
                  {"rdfslot_"})
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
  ;
  // xx
  df.Snapshot("tree", outname,
              {"DM2", "Dm2", "T12", "T13", "T23", "DCP", "count"});

  //   cdata.SaveAs("data.root");

  return 0;
}