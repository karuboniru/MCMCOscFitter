#pragma once
#include "StateI.h"
#include <concepts>

template <typename DataType>
class ModelDataLLH : virtual public StateI {
public:
  [[nodiscard]] virtual double GetLogLikelihoodAgainstData(const DataType &dataset) const = 0;
};

// Concept: Model must derive from some ModelDataLLH<Data>
template <typename Model, typename Data>
concept has_model_data_llh = requires(const Model &m, const Data &d) {
  { m.GetLogLikelihoodAgainstData(d) } -> std::same_as<double>;
};

template <typename Model, typename Data>
  requires has_model_data_llh<Model, Data> && std::derived_from<Model, StateI>
           && std::derived_from<Data, StateI>
class ModelAndData : virtual public StateI {
public:
  ModelAndData(Model model_, Data data_) : model(model_), data(data_) {}
  ModelAndData(const ModelAndData &) = default;
  ModelAndData(ModelAndData &&) = default;
  ModelAndData &operator=(const ModelAndData &) = default;
  ModelAndData &operator=(ModelAndData &&) = default;
  ~ModelAndData() override = default;

  void proposeStep() override {
    model.proposeStep();
    data.proposeStep();
  }

  double GetLogLikelihood() const override {
    auto modeldata = model.GetLogLikelihoodAgainstData(data);
    auto modelonly = model.GetLogLikelihood();
    auto dataonly = data.GetLogLikelihood();
    return modeldata + modelonly + dataonly;
  }

  Model &GetModel() { return model; }
  Data &GetData() { return data; }

private:
  Model model;
  Data data;
};
