#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef CARLOS_USE_OPENMP
#include <omp.h>
#endif

namespace carlos {

inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

class AlignedBuffer {
 public:
  AlignedBuffer() = default;

  explicit AlignedBuffer(size_t count) { allocate(count); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      deallocate();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~AlignedBuffer() { deallocate(); }

  void allocate(size_t count) {
    deallocate();
    size_ = count;
    const size_t bytes = pad_to_cache_line(count * sizeof(double));
    data_ = static_cast<double*>(std::aligned_alloc(64, bytes));
    if (data_ == nullptr) {
      throw std::bad_alloc();
    }
#ifndef NDEBUG
    if (reinterpret_cast<uintptr_t>(data_) % 64 != 0) {
      throw std::runtime_error("AlignedBuffer: pointer not 64-byte aligned");
    }
#endif
  }

  [[nodiscard]] double* data() noexcept { return data_; }
  [[nodiscard]] const double* data() const noexcept { return data_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }

 private:
  static size_t pad_to_cache_line(size_t bytes) {
    constexpr size_t kAlign = 64;
    return ((bytes + kAlign - 1) / kAlign) * kAlign;
  }

  void deallocate() {
    if (data_ != nullptr) {
      std::free(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  double* data_ = nullptr;
  size_t size_ = 0;
};

class GBMSimulator {
 public:
  GBMSimulator(int dim, int num_paths, int num_steps, double dt, double r,
               double T, const std::vector<double>& x0,
               const std::vector<double>& delta, const std::vector<double>& sigma,
               uint64_t seed)
      : dim_(dim),
        num_paths_(num_paths),
        num_steps_(num_steps),
        dt_(dt),
        r_(r),
        T_(T),
        seed_(seed),
        stride_(static_cast<size_t>(num_steps) + 1) {
    if (dim_ <= 0 || num_paths_ <= 0 || num_steps_ <= 0 || dt_ <= 0.0) {
      throw std::invalid_argument("GBMSimulator: invalid dimensions or dt");
    }
    if (static_cast<int>(x0.size()) != dim_ || static_cast<int>(delta.size()) != dim_ ||
        static_cast<int>(sigma.size()) != dim_) {
      throw std::invalid_argument("GBMSimulator: parameter vector size mismatch");
    }

    drift_.resize(static_cast<size_t>(dim_));
    diffusion_.resize(static_cast<size_t>(dim_));
    for (int i = 0; i < dim_; ++i) {
      drift_[static_cast<size_t>(i)] =
          (r_ - delta[static_cast<size_t>(i)] -
           0.5 * sigma[static_cast<size_t>(i)] * sigma[static_cast<size_t>(i)]) *
          dt_;
      diffusion_[static_cast<size_t>(i)] = sigma[static_cast<size_t>(i)] * std::sqrt(dt_);
    }

    x0_ = x0;

    const size_t buf_size = static_cast<size_t>(num_paths_) * stride_;
    paths_.resize(static_cast<size_t>(dim_));
    for (int i = 0; i < dim_; ++i) {
      paths_[static_cast<size_t>(i)].allocate(buf_size);
    }

    rngs_.resize(static_cast<size_t>(num_paths_));
    reseed(seed_);
  }

  void run() { simulate_from(0.0, x0_.data(), num_paths_, seed_); }

  void simulate_from(double t_init, const double* x_init, int num_paths,
                     uint64_t seed) {
    if (x_init == nullptr) {
      throw std::invalid_argument("simulate_from: x_init is null");
    }
    if (num_paths <= 0 || num_paths > num_paths_) {
      throw std::invalid_argument("simulate_from: num_paths out of range");
    }
    if (t_init < 0.0 || t_init > T_) {
      throw std::invalid_argument("simulate_from: t_init out of range");
    }

    const int start_step = static_cast<int>(std::round(t_init / dt_));
    const int steps_remaining = num_steps_ - start_step;
    if (steps_remaining < 0) {
      throw std::invalid_argument("simulate_from: t_init beyond horizon");
    }

    active_paths_ = num_paths;
    active_steps_ = steps_remaining;
    active_start_step_ = start_step;

    reseed(seed, num_paths);

    std::vector<double> init(static_cast<size_t>(dim_));
    for (int i = 0; i < dim_; ++i) {
      init[static_cast<size_t>(i)] = x_init[i];
    }

#ifdef CARLOS_USE_OPENMP
#pragma omp parallel for
    for (int p = 0; p < num_paths; ++p) {
      simulate_path(p, init.data(), start_step, steps_remaining);
    }
#else
    const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
      workers.emplace_back([&, t]() {
        for (int p = static_cast<int>(t); p < num_paths;
             p += static_cast<int>(n_threads)) {
          simulate_path(p, init.data(), start_step, steps_remaining);
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
#endif
  }

  [[nodiscard]] double* data(int dim) {
    if (dim < 0 || dim >= dim_) {
      throw std::out_of_range("data: dim out of range");
    }
    return paths_[static_cast<size_t>(dim)].data();
  }

  [[nodiscard]] const double* data(int dim) const {
    if (dim < 0 || dim >= dim_) {
      throw std::out_of_range("data: dim out of range");
    }
    return paths_[static_cast<size_t>(dim)].data();
  }

  [[nodiscard]] int dim() const noexcept { return dim_; }
  [[nodiscard]] int num_paths() const noexcept { return active_paths_; }
  [[nodiscard]] int num_steps() const noexcept { return active_steps_; }
  [[nodiscard]] int max_paths() const noexcept { return num_paths_; }
  [[nodiscard]] int max_steps() const noexcept { return num_steps_; }
  [[nodiscard]] size_t stride() const noexcept { return stride_; }
  [[nodiscard]] int start_step() const noexcept { return active_start_step_; }
  [[nodiscard]] double dt() const noexcept { return dt_; }
  [[nodiscard]] double T() const noexcept { return T_; }
  [[nodiscard]] double r() const noexcept { return r_; }

 private:
  void reseed(uint64_t seed, int count = -1) {
    const int n = count < 0 ? num_paths_ : count;
    for (int p = 0; p < n; ++p) {
      const uint64_t path_seed = splitmix64(seed + static_cast<uint64_t>(p) * 0x9E3779B97F4A7C15ULL);
      rngs_[static_cast<size_t>(p)].seed(path_seed);
    }
  }

  void simulate_path(int p, const double* x_init, int start_step, int steps_remaining) {
    auto& rng = rngs_[static_cast<size_t>(p)];
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < dim_; ++i) {
      double* buf = paths_[static_cast<size_t>(i)].data();
      buf[static_cast<size_t>(p) * stride_ + static_cast<size_t>(start_step)] = x_init[i];
    }

    for (int n = 0; n < steps_remaining; ++n) {
      const int step = start_step + n;
      for (int i = 0; i < dim_; ++i) {
        double* buf = paths_[static_cast<size_t>(i)].data();
        const size_t idx = static_cast<size_t>(p) * stride_ + static_cast<size_t>(step);
        const size_t next = idx + 1;
        const double z = dist(rng);
        buf[next] = buf[idx] * std::exp(drift_[static_cast<size_t>(i)] +
                                        diffusion_[static_cast<size_t>(i)] * z);
      }
    }
  }

  int dim_;
  int num_paths_;
  int num_steps_;
  double dt_;
  double r_;
  double T_;
  uint64_t seed_;
  size_t stride_;

  int active_paths_ = 0;
  int active_steps_ = 0;
  int active_start_step_ = 0;

  std::vector<double> x0_;
  std::vector<double> drift_;
  std::vector<double> diffusion_;
  std::vector<AlignedBuffer> paths_;
  std::vector<std::mt19937_64> rngs_;
};

}  // namespace carlos
