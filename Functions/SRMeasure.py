import numpy
import math
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import numpy.linalg
from scipy.special import gamma
import scipy.misc
import scipy.io
import skimage.transform



def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

"""
Hat tip: http://stackoverflow.com/a/5078155/1828289
"""
def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)

# FIXME there seems to be a problem with this code
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return numpy.mean(ssim_map)

"""
Generalized Gaussian distribution estimation.
Cite: 
Dominguez-Molina, J. Armando, et al. "A practical procedure to estimate the shape parameter in the generalized Gaussian distribution.", 
  available through http://www. cimat. mx/reportes/enlinea/I-01-18_eng. pdf 1 (2001).
"""

"""
Generalized Gaussian ratio function
Cite: Dominguez-Molina 2001, pg 7, eq (8)
"""


def generalized_gaussian_ratio(alpha):
    return (gamma(2.0 / alpha) ** 2) / (gamma(1.0 / alpha) * gamma(3.0 / alpha))


"""
Generalized Gaussian ratio function inverse (numerical approximation)
Cite: Dominguez-Molina 2001, pg 13
"""


def generalized_gaussian_ratio_inverse(k):
    a1 = -0.535707356
    a2 = 1.168939911
    a3 = -0.1516189217
    b1 = 0.9694429
    b2 = 0.8727534
    b3 = 0.07350824
    c1 = 0.3655157
    c2 = 0.6723532
    c3 = 0.033834

    if k < 0.131246:
        return 2 * math.log(27.0 / 16.0) / math.log(3.0 / (4 * k ** 2))
    elif k < 0.448994:
        return (1 / (2 * a1)) * (-a2 + math.sqrt(a2 ** 2 - 4 * a1 * a3 + 4 * a1 * k))
    elif k < 0.671256:
        return (1 / (2 * b3 * k)) * (b1 - b2 * k - math.sqrt((b1 - b2 * k) ** 2 - 4 * b3 * (k ** 2)))
    elif k < 0.75:
        # print "%f %f %f" % (k, ((3-4*k)/(4*c1)), c2**2 + 4*c3*log((3-4*k)/(4*c1)) )
        return (1 / (2 * c3)) * (c2 - math.sqrt(c2 ** 2 + 4 * c3 * math.log((3 - 4 * k) / (4 * c1))))
    else:
        print("warning: GGRF inverse of %f is not defined" % (k))
        return numpy.nan


"""
Estimate the parameters of an asymmetric generalized Gaussian distribution
"""


def estimate_aggd_params(x):
    x_left = x[x < 0]
    x_right = x[x >= 0]
    stddev_left = math.sqrt((1.0 / (x_left.size - 1)) * numpy.sum(x_left ** 2))
    stddev_right = math.sqrt((1.0 / (x_right.size - 1)) * numpy.sum(x_right ** 2))
    if stddev_right == 0:
        return 1, 0, 0  # TODO check this
    r_hat = numpy.mean(numpy.abs(x)) ** 2 / numpy.mean(x ** 2)
    y_hat = stddev_left / stddev_right
    R_hat = r_hat * (y_hat ** 3 + 1) * (y_hat + 1) / ((y_hat ** 2 + 1) ** 2)
    alpha = generalized_gaussian_ratio_inverse(R_hat)
    beta_left = stddev_left * math.sqrt(gamma(3.0 / alpha) / gamma(1.0 / alpha))
    beta_right = stddev_right * math.sqrt(gamma(3.0 / alpha) / gamma(1.0 / alpha))
    return alpha, beta_left, beta_right


def compute_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)

    features.extend([alpha, (beta_left + beta_right) / 2])

    for x_shift, y_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        img_pair_products = img_norm * numpy.roll(numpy.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0 / alpha) / gamma(1.0 / alpha))
        features.extend([alpha, eta, beta_left, beta_right])

    return features


def normalize_image(img, sigma=7 / 6):
    mu = gaussian_filter(img, sigma, mode='nearest')
    mu_sq = mu * mu
    sigma = numpy.sqrt(numpy.abs(gaussian_filter(img * img, sigma, mode='nearest') - mu_sq))
    img_norm = (img - mu) / (sigma + 1)
    return img_norm


def niqe(img):
    model_mat = scipy.io.loadmat('modelparameters.mat')
    model_mu = model_mat['mu_prisparam']
    model_cov = model_mat['cov_prisparam']

    features = None
    img_scaled = img
    for scale in [1, 2]:

        if scale != 1:
            img_scaled = skimage.transform.rescale(img, 1 / scale)
            # img_scaled = scipy.misc.imresize(img_norm, 0.5)

        # print img_scaled
        img_norm = normalize_image(img_scaled)

        scale_features = []
        block_size = 96 // scale
        for block_col in range(img_norm.shape[0] // block_size):
            for block_row in range(img_norm.shape[1] // block_size):
                block_features = compute_features(img_norm[block_col * block_size:(block_col + 1) * block_size,
                                                  block_row * block_size:(block_row + 1) * block_size])
                scale_features.append(block_features)
        # print "len(scale_features)=%f" %(len(scale_features))
        if features == None:
            features = numpy.vstack(scale_features)
            # print features.shape
        else:
            features = numpy.hstack([features, numpy.vstack(scale_features)])
            # print features.shape

    features_mu = numpy.mean(features, axis=0)
    features_cov = numpy.cov(features.T)

    pseudoinv_of_avg_cov = numpy.linalg.pinv((model_cov + features_cov) / 2)
    niqe_quality = math.sqrt((model_mu - features_mu).dot(pseudoinv_of_avg_cov.dot((model_mu - features_mu).T)))

    return niqe_quality