#include "binning_tool.hpp"

#include <TDirectoryFile.h>
#include <TFile.h>
#include <TH2D.h>

#include <array>
#include <format>
#include <ranges>

const char *const ziou_normal =
    "/var/home/yan/codes/MCMCOscFitter/build/ziou/NO_event.root";
const char *const ziou_inverted =
    "/var/home/yan/codes/MCMCOscFitter/build/ziou/IO_event.root";
const char *const ziou_no_osc =
    "/var/home/yan/codes/MCMCOscFitter/build/ziou/noosc_event.root";
const auto ziou_file_array =
    std::to_array({ziou_normal, ziou_inverted, ziou_no_osc});
const auto ziou_hist_name = std::to_array(
    {"final_numu_flux_rebinned", "final_nue_flux_rebinned",
     "final_numubar_flux_rebinned", "final_nuebar_flux_rebinned"});

const char *const qiyu_normal =
    "/var/home/yan/codes/MCMCOscFitter/build/Event_rate_NH.root";
const char *const qiyu_inverted =
    "/var/home/yan/codes/MCMCOscFitter/build/Event_rate_IH.root";
const char *const qiyu_no_osc =
    "/var/home/yan/codes/MCMCOscFitter/build/No_Osc.root";
const auto qiyu_file_array =
    std::to_array({qiyu_normal, qiyu_inverted, qiyu_no_osc});
const auto qiyu_hist_name = std::to_array({"numu", "nue", "numubar", "nuebar"});

// auto reset_bin (TH2D *hist, const std::vector<double> &x, const
// std::vector<double> &y) {
//     hist->SetBins(x.size() - 1, x.data(), y.size() - 1, y.data());
// }

int main() {
  //   auto costheta_bins = linspace(-1., 1., 11);

  //   auto Ebins = logspace(0.1, 20., 10);
  TFile output = TFile("cross_check.root", "RECREATE");

  for (const auto &[ziou_file, qiyu_file, tag] :
       std::views::zip(ziou_file_array, qiyu_file_array,
                       std::to_array({"normal", "inverted", "no_osc"}))) {
    TFile ziou(ziou_file);
    TFile qiyu(qiyu_file);
    auto dir = output.mkdir(tag);
    dir->cd();
    for (const auto &[ziou_hist_name, qiyu_hist_name] :
         std::views::zip(ziou_hist_name, qiyu_hist_name)) {
      auto ziou_hist = dynamic_cast<TH2D *>(ziou.Get(ziou_hist_name));
      //   ziou_hist->SetBins(Ebins.size() - 1, Ebins.data(),
      //                      costheta_bins.size() - 1, costheta_bins.data());
      auto qiyu_hist = dynamic_cast<TH2D *>(qiyu.Get(qiyu_hist_name));
      ziou_hist->SetBins(qiyu_hist->GetNbinsX(),
                         qiyu_hist->GetXaxis()->GetXbins()->GetArray(),
                         qiyu_hist->GetNbinsY(),
                         qiyu_hist->GetYaxis()->GetXbins()->GetArray());
      ziou_hist->Write(std::format("ziou_{}", qiyu_hist_name).c_str());
      qiyu_hist->Write(std::format("qiyu_{}", qiyu_hist_name).c_str());
      auto hist_diff = dynamic_cast<TH2D *>(
          ziou_hist->Clone(std::format("diff_{}", qiyu_hist_name).c_str()));
      hist_diff->Add(qiyu_hist, -1);
      hist_diff->Write();
      //   auto hist_ratio = dynamic_cast<TH2D *>(
      //       ziou_hist->Clone(std::format("ratio_{}",
      //       qiyu_hist_name).c_str()));
      //   hist_ratio->Divide(qiyu_hist);
      //   hist_ratio->Write();
      auto hist_sum = dynamic_cast<TH2D *>(
          ziou_hist->Clone(std::format("sum_{}", qiyu_hist_name).c_str()));
      hist_sum->Add(qiyu_hist);
      // diff / sum -> ratio
      auto hist_ratio = dynamic_cast<TH2D *>(
          hist_diff->Clone(std::format("ratio_{}", qiyu_hist_name).c_str()));
      hist_ratio->Divide(hist_sum);
      hist_ratio->Write();
    }
    dir->Write();
  }
  output.Write();
  output.Close();
}