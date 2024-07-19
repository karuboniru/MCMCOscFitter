#pragma once
#include "StateI.h"
#include <concepts>
// class DataSet;

class ModelDataLLH : virtual public StateI {
public:
  virtual double GetLogLikelihoodAgainstData(const StateI &dataset) const = 0;
};

template <std::derived_from<ModelDataLLH> Model, std::derived_from<StateI> Data>
class ModelAndData : virtual public StateI {
public:
  ModelAndData(Model model_, Data data_) : model(model_), data(data_) {}

  virtual void proposeStep() override {
    model.proposeStep();
    data.proposeStep();
  }

  virtual double GetLogLikelihood() const override {
    return model.GetLogLikelihoodAgainstData(data) + model.GetLogLikelihood() +
           data.GetLogLikelihood();
  }

  Model & GetModel() { return model; }
  Data & GetData() { return data; }

private:
  Model model;
  Data data;
};