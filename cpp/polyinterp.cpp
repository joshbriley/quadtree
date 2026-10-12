#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Cubic Hermite basis on [lo, lo + h] at normalized t:
// [value at lo, value at hi, slope at lo, slope at hi] (slopes in physical units).
static void hermite_basis(double t, double h, double out[4]) {
    double t2 = t * t;
    double t3 = t2 * t;
    out[0] = 2.0 * t3 - 3.0 * t2 + 1.0;
    out[1] = 3.0 * t2 - 2.0 * t3;
    out[2] = h * (t3 - 2.0 * t2 + t);
    out[3] = h * (t3 - t2);
}

double evaluate_hermite(
    py::array_t<double> vals,
    double x,
    double y,
    py::array_t<double> bounds
) {
    if (vals.ndim() != 2 || vals.shape(0) != 4 || vals.shape(1) != 4) {
        throw py::value_error("vals must be a 4x4 array");
    }
    if (bounds.ndim() != 1 || bounds.shape(0) != 4) {
        throw py::value_error("bounds must be a 1D array of length 4: [xmin, xmax, ymin, ymax]");
    }

    auto v = vals.unchecked<2>();   // fast NumPy access
    auto b = bounds.unchecked<1>();

    double xmin = b(0);
    double xmax = b(1);
    double ymin = b(2);
    double ymax = b(3);

    if (xmax == xmin || ymax == ymin) {
        throw py::value_error("Cell bounds are degenerate");
    }

    double hx = xmax - xmin;
    double hy = ymax - ymin;

    double Hx[4];
    double Hy[4];
    hermite_basis((x - xmin) / hx, hx, Hx);
    hermite_basis((y - ymin) / hy, hy, Hy);

    // Tensor product
    double result = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result += v(i,j) * Hx[i] * Hy[j];
        }
    }

    return result;
}

// Python module name = polyinterp
PYBIND11_MODULE(polyinterp, m) {
    m.def("evaluate_hermite", &evaluate_hermite,
          "Evaluate tensor-product cubic Hermite interpolation");
}
