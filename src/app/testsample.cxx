#include "BargerPropagator.h"
#include "NeutrinoState.h"
#include "OscillationParameters.h"
#include "SimpleDataPoint.h"
#include "SimpleInteraction.h"
#include "StateI.h"
#include "genie_xsec.h"
#include "hondaflux.h"
#include "to_plots.h"
#include "walker.h"

#include "root/Math/IntegratorOptions.h"
#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include <TF3.h>
#include <TMath.h>
#include <TRandom.h>
#include <cmath>
#include <functional>
#include <memory>

class TimeCount {
public:
  TimeCount(std::string name)
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()) {}
  ~TimeCount() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " musec\n";
  }

private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

int main() {
  TH1::AddDirectory(false);
  ROOT::Math::IntegratorMultiDimOptions::SetDefaultIntegrator("MISER");
  const SimpleInteraction interaction;
  size_t e_count{};
  {
    TimeCount tc("weight_int");
    TF3 weight_func(
        "weight_func",
        [&](double *x, double *) {
          double E = exp(x[0]);
          double costh = x[1];
          double phi = x[2];
          NeutrinoState this_state_nue{E, costh, phi, 12};      // nue
          NeutrinoState this_state_nuebar{E, costh, phi, -12};  // nuebar
          NeutrinoState this_state_numu{E, costh, phi, 14};     // numu
          NeutrinoState this_state_numubar{E, costh, phi, -14}; // numubar
          // double logw =
          // interaction.GetLogLikelihoodAgainstData(SimpleDataSet{this_state_nue});
          return E * (exp(interaction.GetLogLikelihoodAgainstData(
                          SimpleDataSet{this_state_nue})) +
                      exp(interaction.GetLogLikelihoodAgainstData(
                          SimpleDataSet{this_state_nuebar})) +
                      exp(interaction.GetLogLikelihoodAgainstData(
                          SimpleDataSet{this_state_numu})) +
                      exp(interaction.GetLogLikelihoodAgainstData(
                          SimpleDataSet{this_state_numubar})));
          // return (weight(E, costh, phi, 14) + weight(E, costh, phi, -14) +
          //         weight(E, costh, phi, -12) + weight(E, costh, phi, 12)) *
          //        E;
        },
        log(Emin), log(Emax), -1, 1, 0, 2 * TMath::Pi(), 0);
    weight_func.SetNpx(50);
    weight_func.SetNpy(50);
    weight_func.SetNpz(30);
    double weight_int = weight_func.Integral(log(Emin), log(Emax), -1, 1, 0,
                                             2 * TMath::Pi(), 0.01);
    e_count = weight_int;
    std::cout << "weight_int: " << weight_int << std::endl;
  }

  // ROOT::EnableImplicitMT();
  NeutrinoState init_state;
  init_state.proposeStep();
  auto df =
      ROOT::RDataFrame{e_count}
          .Define("state",
                  [&]() {
                    for (size_t i{}; i < 200; i++) {
                      NeutrinoState newState = init_state;
                      newState.proposeStep();
                      double logw = interaction.GetLogLikelihoodAgainstData(
                          SimpleDataSet{newState});
                      double logw0 = interaction.GetLogLikelihoodAgainstData(
                          SimpleDataSet{init_state});
                      double delta = logw - logw0;
                      if (delta > 0) {
                        init_state = newState;
                      } else {
                        auto rand = gRandom->Rndm();
                        if (rand < exp(delta)) {
                          init_state = newState;
                        }
                      }
                    }
                    return init_state;
                  })
          .Define("E", [](const NeutrinoState &s) { return s.E; }, {"state"})
          .Define("costheta", [](const NeutrinoState &s) { return s.costheta; },
                  {"state"})
          .Define("phi", [](const NeutrinoState &s) { return s.phi; },
                  {"state"})
          .Define("w",
                  [&](const NeutrinoState &s) {
                    return exp(interaction.GetLogLikelihoodAgainstData(
                        SimpleDataSet{s}));
                  },
                  {"state"})
          .Define("flavor", [](const NeutrinoState &s) { return s.flavor; },
                  {"state"})
          .Define("weight", []() { return 1.; }, {});
  ROOT::RDF::Experimental::AddProgressBar(df);
  auto plots = to_plots(df);
  df.Snapshot("testevent", "testsample.root",
              {"E", "costheta", "phi", "flavor", "w"});
  auto filehist = std::make_unique<TFile>("testsample.root", "RECREATE");

  for (auto &&plot : plots) {
    filehist->Add(plot.GetPtr());
  }
  filehist->Write();
  filehist->Close();

  return 0;
}