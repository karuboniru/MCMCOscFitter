#pragma once
#include "mcmc_concepts.h"
#include <concepts>

// Concept: Model must provide GetLogLikelihoodAgainstData(Data) → double.
// This replaces the old ModelDataLLH<DataType> class template.
template <typename Model, typename Data>
concept has_model_data_llh = requires(const Model &m, const Data &d) {
  { m.GetLogLikelihoodAgainstData(d) } -> std::same_as<double>;
};

// ModelAndData composes a model and a data set for MCMC sampling.
// Concept-constrained: each of M and D must satisfy MCMCState, and M must
// additionally satisfy has_model_data_llh<M, D>.
//
// No virtual inheritance — the class itself satisfies MCMCState by delegating
// proposeStep() and GetLogLikelihood() to its members.
template <typename M, typename D>
  requires has_model_data_llh<M, D> && mcmc_concepts::MCMCState<M>
           && mcmc_concepts::MCMCState<D>
class ModelAndData {
public:
  ModelAndData(M model_, D data_) : model(model_), data(data_) {}
  ModelAndData(const ModelAndData &) = default;
  ModelAndData(ModelAndData &&) = default;
  ModelAndData &operator=(const ModelAndData &) = default;
  ModelAndData &operator=(ModelAndData &&) = default;

  void proposeStep() {
    model.proposeStep();
    data.proposeStep();
  }

  double GetLogLikelihood() const {
    auto modeldata = model.GetLogLikelihoodAgainstData(data);
    auto modelonly = model.GetLogLikelihood();
    auto dataonly = data.GetLogLikelihood();
    return modeldata + modelonly + dataonly;
  }

  M &GetModel() { return model; }
  const M &GetModel() const { return model; }
  D &GetData() { return data; }
  const D &GetData() const { return data; }

private:
  M model;
  D data;
};
