
import numpy as np # type: ignore
from scipy.stats.mstats import zscore
import math

__all__ = [
    "compute_correlation",
]


def _normalize_for_correlation(data, axis, return_nans=False):
    """normalize the data before computing correlation

    The data will be z-scored and divided by sqrt(n)
    along the assigned axis

    Parameters
    ----------
    data: 2D array

    axis: int
        specify which dimension of the data should be normalized

    return_nans: bool, default:False
        If False, return zeros for NaNs; if True, return NaNs

    Returns
    -------
    data: 2D array
        the normalized data
    """
    shape = data.shape
    data = zscore(data, axis=axis, ddof=0)
    # if zscore fails (standard deviation is zero),
    # optionally set all values to be zero
    if not return_nans:
        data = np.nan_to_num(data)
    data = data / math.sqrt(shape[axis])
    return data


def compute_correlation(matrix1, matrix2, return_nans=False):
    """compute correlation between two sets of variables

    Correlate the rows of matrix1 with the rows of matrix2.
    If matrix1 == matrix2, it is auto-correlation computation
    resulting in a symmetric correlation matrix.
    The number of columns MUST agree between set1 and set2.
    The correlation being computed here is
    the Pearson's correlation coefficient, which can be expressed as

    .. math:: corr(X, Y) = \\frac{cov(X, Y)}{\\sigma_X\\sigma_Y}

    where cov(X, Y) is the covariance of variable X and Y, and

    .. math:: \\sigma_X

    is the standard deviation of variable X

    Reducing the correlation computation to matrix multiplication
    and using BLAS GEMM API wrapped by Scipy can speedup the numpy built-in
    correlation computation (numpy.corrcoef) by one order of magnitude

    .. math::
        corr(X, Y)
        &= \\frac{\\sum\\limits_{i=1}^n (x_i-\\bar{x})(y_i-\\bar{y})}{(n-1)
        \\sqrt{\\frac{\\sum\\limits_{j=1}^n x_j^2-n\\bar{x}}{n-1}}
        \\sqrt{\\frac{\\sum\\limits_{j=1}^{n} y_j^2-n\\bar{y}}{n-1}}}\\\\
        &= \\sum\\limits_{i=1}^n(\\frac{(x_i-\\bar{x})}
        {\\sqrt{\\sum\\limits_{j=1}^n x_j^2-n\\bar{x}}}
        \\frac{(y_i-\\bar{y})}{\\sqrt{\\sum\\limits_{j=1}^n y_j^2-n\\bar{y}}})

    By default (return_nans=False), returns zeros for vectors with NaNs.
    If return_nans=True, convert zeros to NaNs (np.nan) in output.

    Parameters
    ----------
    matrix1: 2D array in shape [r1, c]
        MUST be continuous and row-major

    matrix2: 2D array in shape [r2, c]
        MUST be continuous and row-major

    return_nans: bool, default:False
        If False, return zeros for NaNs; if True, return NaNs

    Returns
    -------
    corr_data: 2D array in shape [r1, r2]
        continuous and row-major in np.float32
    """
    matrix1 = matrix1.astype(np.float32)
    matrix2 = matrix2.astype(np.float32)
    [r1, d1] = matrix1.shape
    [r2, d2] = matrix2.shape
    if d1 != d2:
        raise ValueError('Dimension discrepancy')
    # preprocess two components
    matrix1 = _normalize_for_correlation(matrix1, 1,
                                         return_nans=return_nans)
    matrix2 = _normalize_for_correlation(matrix2, 1,
                                         return_nans=return_nans)
    corr_data = np.empty((r1, r2), dtype=np.float32, order='C')
    # blas routine is column-major
    blas.compute_single_matrix_multiplication('T', 'N',
                                              r2, r1, d1,
                                              1.0,
                                              matrix2, d2,
                                              matrix1, d1,
                                              0.0,
                                              corr_data,
                                              r2)
    return corr_data
