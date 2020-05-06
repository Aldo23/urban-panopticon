import cv2
import imutils
import numpy as np


def four_point_transform(image, pts):
    pts = pts.astype(dtype='float32')
    (tl, bl, br, tr) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0]], dtype="float32")

    # calculate matrix H
    h, status = cv2.findHomography(pts, dst)

    # print('\n Homography matrix')
    # print(h)

    warped = cv2.warpPerspective(image, h, (width, height))
    return warped, h


def four_point_transform_geo(image, pts, loc):
    pts = pts.astype(dtype='float32')
    (tl, bl, br, tr) = pts

    dst = np.array([
        [loc.iloc[0]['X'], loc.iloc[0]['Y']],
        [loc.iloc[1]['X'], loc.iloc[1]['Y']],
        [loc.iloc[2]['X'], loc.iloc[2]['Y']],
        [loc.iloc[3]['X'], loc.iloc[3]['Y']]], dtype="float32")

    h, status = cv2.findHomography(pts, dst)

    warped = cv2.warpPerspective(image, h, (image.shape[1], image.shape[0]))
    return warped, h


def find_markers(image):
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([40, 255, 40])
    mask = cv2.inRange(image, lower_green, upper_green)
    # cv2.imshow('marked', mask)
    # cv2.waitKey(0)
    res = cv2.bitwise_and(image, image, mask=mask)
    blurred = cv2.GaussianBlur(res, (11, 11), cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = imutils.resize(thresh, width=image.shape[1])
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centroids = np.asarray([[0, 0]])[1:]
    for ind, c in enumerate(cnts):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        markers = np.asarray([[cX, cY]])
        centroids = np.concatenate((centroids, markers), axis=0)
    #     cv2.putText(image, str(ind), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # cv2.imshow('marked', image)
    # cv2.waitKey(0)
    return centroids


def sort_markers(markers, image):
    points = markers
    center = points.mean(axis=0)
    moved = points - center
    r = np.linalg.norm(moved, axis=1)
    # print (r)
    y = moved[:, 1]
    x = moved[:, 0]
    arccos = np.arccos(y/r)
    sign = np.where(x >= 0, 1, -1)
    theta = arccos * sign
    key = theta.argsort()
    ordered = np.zeros((len(key), 2), dtype=int)
    for i in range(len(key)):
        ordered[i] = points[key[i]]
    for ind, center in enumerate(ordered):
        cv2.putText(image, str(
            ind), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.imshow("Original", image)
    # cv2.waitKey(0)
    return ordered, image


def warpPerspectivePadded(
        src, dst, M,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0):
    """Performs a perspective warp with padding.
    Parameters
    ----------
    src : array_like
        source image, to be warped.
    dst : array_like
        destination image, to be padded.
    M : array_like
        `3x3` perspective transformation matrix.
    Returns
    -------
    src_warped : ndarray
        padded and warped source image
    dst_padded : ndarray
        padded destination image, same size as src_warped
    Optional Parameters
    -------------------
    flags : int, optional
        combination of interpolation methods (`cv2.INTER_LINEAR` or
        `cv2.INTER_NEAREST`) and the optional flag `cv2.WARP_INVERSE_MAP`,
        that sets `M` as the inverse transformation (`dst` --> `src`).
    borderMode : int, optional
        pixel extrapolation method (`cv2.BORDER_CONSTANT` or
        `cv2.BORDER_REPLICATE`).
    borderValue : numeric, optional
        value used in case of a constant border; by default, it equals 0.
    See Also
    --------
    warpAffinePadded() : for `2x3` affine transformations
    cv2.warpPerspective(), cv2.warpAffine() : original OpenCV functions
    """

    assert M.shape == (3, 3), \
        'Perspective transformation shape should be (3, 3).\n' \
        + 'Use warpAffinePadded() for (2, 3) affine transformations.'

    M = M / M[2, 2]  # ensure a legal homography
    if flags in (cv2.WARP_INVERSE_MAP,
                 cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                 cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP):
        M = cv2.invert(M)[1]
        flags -= cv2.WARP_INVERSE_MAP

    # it is enough to find where the corners of the image go to find
    # the padding bounds; points in clockwise order from origin
    src_h, src_w = src.shape[:2]
    lin_homg_pts = np.array([
        [0, src_w, src_w, 0],
        [0, 0, src_h, src_h],
        [1, 1, 1, 1]])

    # transform points
    transf_lin_homg_pts = M.dot(lin_homg_pts)
    transf_lin_homg_pts /= transf_lin_homg_pts[2, :]

    # find min and max points
    min_x = np.floor(np.min(transf_lin_homg_pts[0])).astype(int)
    min_y = np.floor(np.min(transf_lin_homg_pts[1])).astype(int)
    max_x = np.ceil(np.max(transf_lin_homg_pts[0])).astype(int)
    max_y = np.ceil(np.max(transf_lin_homg_pts[1])).astype(int)

    # add translation to the transformation matrix to shift to positive values
    anchor_x, anchor_y = 0, 0
    transl_transf = np.eye(3, 3)
    if min_x < 0:
        anchor_x = -min_x
        transl_transf[0, 2] += anchor_x
    if min_y < 0:
        anchor_y = -min_y
        transl_transf[1, 2] += anchor_y
    shifted_transf = transl_transf.dot(M)
    shifted_transf /= shifted_transf[2, 2]

    # create padded destination image
    dst_h, dst_w = dst.shape[:2]

    pad_widths = [anchor_y, max(max_y, dst_h) - dst_h,
                  anchor_x, max(max_x, dst_w) - dst_w]

    dst_padded = cv2.copyMakeBorder(dst, *pad_widths,
                                    borderType=borderMode, value=borderValue)

    dst_pad_h, dst_pad_w = dst_padded.shape[:2]
    src_warped = cv2.warpPerspective(
        src, shifted_transf, (dst_pad_w, dst_pad_h),
        flags=flags, borderMode=borderMode, borderValue=borderValue)

    return src_warped, shifted_transf
