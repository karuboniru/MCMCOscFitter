#include <chrono>
#include <iostream>
#include <string>

class TimeCount {
public:
  TimeCount(std::string name)
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()) {}
  TimeCount(TimeCount const &) = delete;
  TimeCount &operator=(TimeCount const &) = delete;
  TimeCount(TimeCount &&) = delete;
  TimeCount &operator=(TimeCount &&) = delete;
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