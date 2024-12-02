#include "binning_tool.hpp"
#include <TFile.h>
#include <TH2D.h>
#include <array>
#include <format>
#include <ranges>

const char *file_ziou =
    "/var/home/yan/.var/app/com.tencent.WeChat/xwechat_files/yanqiyu1_074d/msg/"
    "file/2024-12/osc_prob/merged.root";
auto ziou_hist_name = std::to_array({"qiyu_nb_invert/NuEToNuE3f", "qiyu_nb_invert/NuMuToNuE3f"});

const char *file_qiyu =
    "/var/home/yan/code/MCMCOscFitter/build/src/app/IH.root";
auto qiyu_hist_name =
    std::to_array({"antineutrino_nue_nue", "antineutrino_nue_numu"});

int main() {
  TFile output = TFile("cross_check_prob.root", "RECREATE");

  TFile ziou(file_ziou);
  TFile qiyu(file_qiyu);
  auto dir = output.mkdir("normal");
  dir->cd();
  for (const auto &[ziou_hist_name, qiyu_hist_name] :
       std::views::zip(ziou_hist_name, qiyu_hist_name)) {
    auto ziou_hist = dynamic_cast<TH2D *>(ziou.Get(ziou_hist_name));

    auto qiyu_hist = dynamic_cast<TH2D *>(qiyu.Get(qiyu_hist_name));
    ziou_hist->Write(std::format("ziou_{}", qiyu_hist_name).c_str());
    qiyu_hist->Write(std::format("qiyu_{}", qiyu_hist_name).c_str());
    auto hist_diff = dynamic_cast<TH2D *>(
        ziou_hist->Clone(std::format("diff_{}", qiyu_hist_name).c_str()));
    hist_diff->Add(qiyu_hist, -1);
    hist_diff->Write();
    auto hist_sum = dynamic_cast<TH2D *>(
        ziou_hist->Clone(std::format("sum_{}", qiyu_hist_name).c_str()));
    hist_sum->Add(qiyu_hist);
    // diff / sum -> ratio
    auto hist_ratio = dynamic_cast<TH2D *>(
        hist_diff->Clone(std::format("ratio_{}", qiyu_hist_name).c_str()));
    hist_ratio->Divide(hist_sum);
    hist_ratio->Write();
  }
  output.Close();
}