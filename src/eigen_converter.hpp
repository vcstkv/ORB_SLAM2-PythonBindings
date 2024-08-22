#ifndef H_EIGEN_TO_NUMPY_CONVERTER
#define H_EIGEN_TO_NUMPY_CONVERTER

#include <Eigen/Core>
#include <boost/python.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

template <typename VectorType>
static PyObject* convert_eigen_vector(const VectorType& mat)
{
    static_assert(VectorType::ColsAtCompileTime == 1 || VectorType::RowsAtCompileTime == 1, "Passed a Matrix into a Vector generator"); // Ensure that it is a vector

    np::dtype dt = np::dtype::get_builtin<typename VectorType::Scalar>();
    auto shape = py::make_tuple(mat.size());
    np::ndarray mOut = np::empty(shape, dt);

    for (Eigen::Index i = 0; i < mat.size(); ++i)
        mOut[i] = mat(i);

    return py::incref(mOut.ptr());
}

template <typename MatrixType>
static PyObject* convert_eigen_matrix(const MatrixType& mat)
{
    static_assert(MatrixType::ColsAtCompileTime != 1 && MatrixType::RowsAtCompileTime != 1, "Passed a Vector into a Matrix generator"); // Ensure that it is not a vector

    np::dtype dt = np::dtype::get_builtin<typename MatrixType::Scalar>();
    auto shape = py::make_tuple(mat.rows(), mat.cols());
    np::ndarray mOut = np::empty(shape, dt);

    for (Eigen::Index i = 0; i < mat.rows(); ++i)
        for (Eigen::Index j = 0; j < mat.cols(); ++j)
            mOut[i][j] = mat(i, j);

    return py::incref(mOut.ptr());
}

#endif // H_EIGEN_TO_NUMPY_CONVERTER
