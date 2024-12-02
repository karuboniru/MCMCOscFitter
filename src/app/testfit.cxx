#include "BargerPropagator.h"
#include "SimpleInteraction.h"
#include "StateI.h"
#include "genie_xsec.h"
#include "hondaflux.h"
#include "walker.h"

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include <TMath.h>
#include <TRandom.h>
#include <cmath>

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
  SimpleInteraction model{};
  model.proposeStep();
  SimpleDataSet dataset{};

  {
    ROOT::RDataFrame data{
        "testevent",
        "/var/home/yan/code/MCMCOscFitter/build/src/app/testsample.root"};
    data.Foreach(
        [&dataset](double E, double costh, double phi, int flavor) {
          dataset.emplace_back(E, costh, phi, flavor);
        },
        {"E", "costheta", "phi", "flavor"});
  }
  std::cerr << "dataset size: " << dataset.size() << std::endl;
  ModelAndData comp(model, dataset);
  

  ROOT::RDataFrame{10000}
      .Define("tuple",
              [&]() {
                TimeCount time_count("iteration");
                auto comp_new = comp;
                comp_new.proposeStep();
                if (MCMCAcceptState(comp, comp_new)) {
                  comp = comp_new;
                }
                // std::cerr << "DM2: " << comp.GetModel().GetDM32sq()
                //           << " T23: " << comp.GetModel().GetT23() << std::endl;
                return std::make_tuple(comp.GetModel().GetDM32sq(),
                                       comp.GetModel().GetT23());
              },
              {})
      .Define("DM2",
              [](std::tuple<double, double> &t) { return std::get<0>(t); },
              {"tuple"})
      .Define("T23",
              [](std::tuple<double, double> &t) { return std::get<1>(t); },
              {"tuple"})
      .Snapshot("tree", "testfit.root", {"DM2", "T23"});

  // TFile file{"testfit.root", "RECREATE"};
  // TTree tree{"tree", "tree"};
  // double DM2;
  // tree.Branch("DM2", &DM2, "DM2/D");

  // comp.proposeStep();
  // for (int i = 0; i < 1000; ++i) {
  //   std::cerr << "iteration: " << i << std::endl;
  //   auto comp_new = comp;
  //   comp_new.proposeStep();
  //   if(MCMCAcceptState(comp, comp_new)) {
  //     comp = comp_new;
  //   }
  //   DM2 = comp.GetModel().GetDM32sq();
  //   std::cerr << "DM2: " << DM2 << std::endl;
  //   tree.Fill();
  // }
  // tree.SaveAs("testfit.root");

  return 0;
}