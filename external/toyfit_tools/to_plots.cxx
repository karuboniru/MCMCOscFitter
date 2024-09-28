#include "to_plots.h"
#include "NeutrinoState.h"

#include <TDatabasePDG.h>

ROOT::RDF::RResultPtr<TH2D> get_plot(ROOT::RDF::RNode rdf_in, int flavor) {
  auto neutrino_name = TDatabasePDG::Instance()->GetParticle(flavor)->GetName();
  return rdf_in.Filter([flavor](int f) { return f == flavor; }, {"flavor"})
      .Histo2D({neutrino_name, neutrino_name, 20, Emin, Emax, 25, -1, 1}, "E",
               "costheta");
}
std::array<ROOT::RDF::RResultPtr<TH2D>, 4> to_plots(ROOT::RDF::RNode df_in) {
  return {get_plot(df_in, 14), get_plot(df_in, -14), get_plot(df_in, 12),
          get_plot(df_in, -12)};
}