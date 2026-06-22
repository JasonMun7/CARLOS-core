#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "simulation.hpp"

namespace py = pybind11;

using SimPtr = std::shared_ptr<carlos::GBMSimulator>;

namespace {

py::array_t<double> paths_numpy_view(const SimPtr& sim, int dim) {
  if (dim < 0 || dim >= sim->dim()) {
    throw py::value_error("dim out of range");
  }
  const int n_paths = sim->num_paths();
  const int n_steps = sim->num_steps();
  const int start = sim->start_step();
  const size_t stride = sim->stride();

  double* base = sim->data(dim) + static_cast<size_t>(start);
  std::vector<py::ssize_t> shape = {n_paths, n_steps + 1};
  std::vector<py::ssize_t> strides = {
      static_cast<py::ssize_t>(stride * sizeof(double)),
      static_cast<py::ssize_t>(sizeof(double))};

  return py::array_t<double>(shape, strides, base, py::cast(sim));
}

py::array_t<double> paths_all_copy(const SimPtr& sim) {
  const int d = sim->dim();
  const int n_paths = sim->num_paths();
  const int n_steps = sim->num_steps();
  const int start = sim->start_step();
  const size_t stride = sim->stride();

  py::array_t<double> arr({d, n_paths, n_steps + 1});
  double* dst = arr.mutable_data();

  for (int i = 0; i < d; ++i) {
    const double* src = sim->data(i);
    for (int p = 0; p < n_paths; ++p) {
      for (int s = 0; s <= n_steps; ++s) {
        const size_t idx =
            (static_cast<size_t>(i) * static_cast<size_t>(n_paths) +
             static_cast<size_t>(p)) *
                static_cast<size_t>(n_steps + 1) +
            static_cast<size_t>(s);
        dst[idx] =
            src[static_cast<size_t>(p) * stride + static_cast<size_t>(start + s)];
      }
    }
  }
  return arr;
}

SimPtr make_simulator(int dim, int num_paths, int num_steps, double dt, double r,
                      double T, const std::vector<double>& x0,
                      const std::vector<double>& delta, const std::vector<double>& sigma,
                      uint64_t seed) {
  return std::make_shared<carlos::GBMSimulator>(dim, num_paths, num_steps, dt, r, T, x0,
                                                delta, sigma, seed);
}

}  // namespace

PYBIND11_MODULE(_carlos_sim, m) {
  m.doc() = "CARLOS GBM Monte Carlo simulator (zero-copy SoA paths)";

  py::class_<carlos::GBMSimulator, SimPtr>(m, "GBMSimulator")
      .def(py::init(&make_simulator), py::arg("dim"), py::arg("num_paths"),
           py::arg("num_steps"), py::arg("dt"), py::arg("r"), py::arg("T"), py::arg("x0"),
           py::arg("delta"), py::arg("sigma"), py::arg("seed"))
      .def("run", &carlos::GBMSimulator::run)
      .def(
          "simulate_from",
          [](SimPtr& sim, double t_init, const std::vector<double>& x_init, int num_paths,
             uint64_t seed) {
            if (static_cast<int>(x_init.size()) != sim->dim()) {
              throw std::invalid_argument("x_init size must match dim");
            }
            sim->simulate_from(t_init, x_init.data(), num_paths, seed);
          },
          py::arg("t_init"), py::arg("x_init"), py::arg("num_paths"), py::arg("seed"))
      .def("paths", &paths_numpy_view, py::arg("dim") = 0,
           "Zero-copy view of paths for asset dimension dim. Invalid after simulator is destroyed.")
      .def("paths_all", &paths_all_copy,
           "Copy of all dimensions as (d, num_paths, num_steps+1).")
      .def_property_readonly("dim", &carlos::GBMSimulator::dim)
      .def_property_readonly("num_paths", &carlos::GBMSimulator::num_paths)
      .def_property_readonly("num_steps", &carlos::GBMSimulator::num_steps)
      .def_property_readonly("dt", &carlos::GBMSimulator::dt)
      .def_property_readonly("T", &carlos::GBMSimulator::T)
      .def_property_readonly("r", &carlos::GBMSimulator::r);
}
