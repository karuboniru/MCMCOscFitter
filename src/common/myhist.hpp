#pragma once

#include <stdexcept>
#include <utility>
#include <valarray>
#include <vector>

template <typename T, size_t dimension>
class MyHistND : public std::valarray<T> {
public:
  using array_type = std::valarray<T>;
  using value_type = T;
  constexpr static size_t Dimension = dimension;

  template <typename... Args>
  MyHistND(Args &&...bin_edge_lists)
      : bin_edges{std::forward<decltype(bin_edge_lists)>(bin_edge_lists)...},
        std::valarray<T>{[&]() {
          size_t size = 1;
          for (size_t i = 0; i < dimension; i++) {
            size *= bin_edges[i].size() - 1;
          }
          return size;
        }()} {
    for (size_t i = 0; i < dimension; i++) {
      if (bin_edges[i].size() < 2) {
        throw std::length_error("bin_edges.size() < 2");
      }
    }
  }
  MyHistND(const MyHistND &other) = default;
  MyHistND(MyHistND &&other) noexcept = default;
  MyHistND &operator=(const MyHistND &other) = default;
  MyHistND &operator=(MyHistND &&other) noexcept = default;
  ~MyHistND() = default;

private:
  std::array<std::vector<double>, dimension> bin_edges;
};


