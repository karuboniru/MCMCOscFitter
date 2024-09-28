#pragma once
#include "StateI.h"
#include <concepts>
// #include <iostream>

// class DataSet;

class ModelDataLLH : virtual public StateI {
public:
  virtual double GetLogLikelihoodAgainstData(const StateI &dataset) const = 0;
};

template <std::derived_from<ModelDataLLH> Model, std::derived_from<StateI> Data>
class ModelAndData : virtual public StateI {
public:
  ModelAndData(Model model_, Data data_) : model(model_), data(data_) {}
  ModelAndData(const ModelAndData &) = default;
  ModelAndData(ModelAndData &&) = default;
  ModelAndData &operator=(const ModelAndData &) = default;
  ModelAndData &operator=(ModelAndData &&) = default;
  ~ModelAndData() = default;

  void proposeStep() override {
    model.proposeStep();
    data.proposeStep();
  }

  double GetLogLikelihood() const override {
    auto modeldata = model.GetLogLikelihoodAgainstData(data);
    auto modelonly = model.GetLogLikelihood();
    auto dataonly = data.GetLogLikelihood();
    // std::cout << std::format("modeldata: {:.2f}\tmodelonly: {:.2f}\tdataonly:
    // {:.2f}\n",
    //                          modeldata, modelonly, dataonly);
    return modeldata + modelonly + dataonly;
  }

  Model &GetModel() { return model; }
  Data &GetData() { return data; }

private:
  Model model;
  Data data;
};