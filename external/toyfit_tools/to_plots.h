#pragma once
#include "NeutrinoState.h"

#include <ROOT/RResultPtr.hxx>
#include <array>

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>

#include <TH2D.h>

std::array<ROOT::RDF::RResultPtr<TH2D>, 4> to_plots(ROOT::RDF::RNode);
