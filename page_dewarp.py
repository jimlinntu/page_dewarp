#!/usr/bin/env python
######################################################################
# page_dewarp.py - Proof-of-concept of page-dewarping based on a
# "cubic sheet" model. Requires OpenCV (version 3 or greater),
# PIL/Pillow, and scipy.optimize.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: MIT License (see LICENSE.txt)
######################################################################

import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
import scipy.optimize

# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101

PAGE_MARGIN_X = 50       # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 20       # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour

EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
FOCAL_LENGTH = 1.2       # normalized focal length of camera

DEBUG_LEVEL = 0          # 0=none, 1=some, 2=lots, 3=all
DEBUG_OUTPUT = 'file'    # file, screen, both

WINDOW_NAME = 'Dewarp'   # Window name for visualization

# nice color palette for visualizing contours, etc.
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)


def debug_show(name, step, text, display):

    if DEBUG_OUTPUT != 'screen':
        filetext = text.replace(' ', '_')
        outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != 'file':

        image = display.copy()
        height = image.shape[0]

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)

        while cv2.waitKey(5) < 0:
            pass


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem


def pix2norm(shape, pts):
    assert isinstance(pts, np.ndarray) and pts.shape[-1] == 2 # (x,y)
    # Normalized pts of image coordinates to coordinates that centered at the center of image
    height, width = shape[:2]
    scl = 2.0/(max(height, width)) # 2 is for mapping [-0.5, 0.5] to [-1.0, 1.0], max(height, width) is to normalize all axis in [0, 1]
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5

    # Each (x,y) will fall into [-1, 1]
    return (pts - offset) * scl


def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def fltp(point):
    return tuple(point.astype(int).flatten())


def draw_correspondences(img, dstpoints, projpts):

    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 1, cv2.LINE_AA)

    return display


def get_default_params(corners, ycoords, xcoords):
    '''
        corners:
            [0]----------[1]
             |            |
             |            |
             |            |
             |            |
             |            |
            [3]----------[2]
        
    '''

    # page width and height
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (page_width, page_height) # Rotated page rough dimension

    # our initial guess for the cubic has no slope
    cubic_slopes = [0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])

    # estimate rotation and translation from four 2D-to-3D point (it is an initial guess from only 4 points)
    # correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))

    span_counts = [len(xc) for xc in xcoords]

    '''
        params[0:3] == rotation vector
        params[3:6] == translation vector
        params[6:8] == cubic_slopes
        params[8:] == ycoords
    '''
    assert len(ycoords) ==  len(xcoords) == len(span_counts)
    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) +
                       tuple(xcoords))

    return rough_dims, span_counts, params


def project_xy(xy_coords, pvec):

    # get cubic polynomial coefficients given
    #
    #  f(0) = 0, f'(0) = alpha
    #  f(1) = 0, f'(1) = beta

    alpha, beta = tuple(pvec[CUBIC_IDX])

    poly = np.array([
        alpha + beta,
        -2*alpha - beta,
        alpha,
        0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        K, np.zeros(5))

    return image_points


def project_keypoints(pvec, keypoint_index):

    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0 # (0,0,0) should be projected to corners[0]

    return project_xy(xy_coords, pvec)


def resize_to_screen(src, maxw=1280, maxh=700, copy=False):

    height, width = src.shape[:2]

    scl_x = float(width)/maxw
    scl_y = float(height)/maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0/scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def get_page_extents(small):

    height, width = small.shape[:2]

    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width-PAGE_MARGIN_X
    ymax = height-PAGE_MARGIN_Y

    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


def get_mask(name, small, pagemask, masktype):

    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    if masktype == 'text':

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     25)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.1, 'thresholded', mask)

        mask = cv2.dilate(mask, box(9, 1))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.2, 'dilated', mask)

        mask = cv2.erode(mask, box(1, 3))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.3, 'eroded', mask)

    else:

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     7)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(8, 2))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.6, 'dilated', mask)

    return np.minimum(mask, pagemask)


def interval_measure_overlap(int_a, int_b):
    # The distance of [max among minimums, min among maximums]
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


def angle_dist(angle_b, angle_a):

    diff = angle_b - angle_a

    while diff > np.pi:
        diff -= 2*np.pi

    while diff < -np.pi:
        diff += 2*np.pi

    return np.abs(diff)


def blob_mean_and_tangent(contour):

    moments = cv2.moments(contour)

    area = moments['m00'] # Note that this computed by Green theroem, so it will probably differ from the result of drawContours

    mean_x = moments['m10'] / area
    mean_y = moments['m01'] / area

    # A covariance matrix over all datapoints (contour surrounded regions)
    moments_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area

    # Perform PCA using SVD
    _, svd_u, _ = cv2.SVDecomp(moments_matrix)

    center = np.array([mean_x, mean_y])
    # Use the principle axis as the main direction of this contour object
    tangent = svd_u[:, 0].flatten().copy()
    # After testing, I found that tangent.x will always be > 0 empirically.
    # But I think it will be better checking the tangent.x direction to gaurantee point0 is smaller and point1 is bigger in x
    tangent = (-1 if tangent[0] < 0 else 1) * tangent # normalize the x coordinate

    return center, tangent


class ContourInfo(object):

    def __init__(self, contour, rect, mask):

        self.contour = contour
        self.rect = rect
        self.mask = mask # 1 indicates the contour's inner region

        self.center, self.tangent = blob_mean_and_tangent(contour)

        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        # Local expansion range
        self.local_xrng = (lxmin, lxmax)

        # Use the "farest" projection as this contour's expansion interval
        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    # I think the author name it as proj_x is because the text region mainly expand in x axis
    def proj_x(self, point):
        # Compute |x||tangent|cos theta = |x| cos theta, which is the signed length of x vector projected on self.tangent
        return np.dot(self.tangent, point.flatten()-self.center)

    # Basically, this function will evalutate "parallelism" of two contours,
    # if two contours is contain too much parallelism, they will have higher score
    def local_overlap(self, other):
        xmin = self.proj_x(other.point0) # Other point's projection to the principal axis (length + direction)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def generate_candidate_edge(cinfo_a, cinfo_b):

    # we want a left of b (so a's successor will be b and b's
    # predecessor will be a) make sure right endpoint of b is to the
    # right of left endpoint of a.
    if cinfo_a.point0[0] > cinfo_b.point1[0]: # contour a's minimum x coordinate must be smaller than contour b's largest
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp

    # Use the mass center of cinfo_a to evaluate the score of cinfo_b's point0 and point1 principal component score
    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    # The tangent vector between two contour center from left point to right
    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

    '''
        * delta_angle:

        Case 1: b is on the top left of a
        y
        ------> x
        |
        v
            -----------
            |    b    |
            -----------

                   --------------
                   |      a     |
                   --------------

            Then overall_angle (the angle of vector a -> b) will be very large (positive)

        Case 2: a is on the left of b: so the angle of vector a -> b will be very small
            -----------     --------------
            |    a    |     |      b     |
            -----------     --------------
    '''
    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi

    # we want the largest overlap in x to be small
    x_overlap = max(x_overlap_a, x_overlap_b)

    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

    # If either their distance is too far or overlap in x is too large or there angle is too different,
    # we do not consider the two contours in the same span
    if (dist > EDGE_MAX_LENGTH or
            x_overlap > EDGE_MAX_OVERLAP or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        # Give a score of the relationship between two contours
        score = dist + delta_angle*EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b) # cinfo_a: left, cino_b: right


def make_tight_mask(contour, xmin, ymin, width, height):

    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)

    return tight_mask


def get_contours(name, small, pagemask, masktype):

    mask = get_mask(name, small, pagemask, masktype)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    contours_out = []

    for contour in contours:

        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        # if this region widht or height is too small and its aspect is too small, then we throw away this region
        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT*height): # equivalent to width / hegith < TEXT_MIN_ASPECT:w
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    if DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)

    return contours_out


def assemble_spans(name, small, pagemask, cinfo_list):

    # sort the contour info list by the value of ymin
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])

    # generate all candidate edges
    candidate_edges = []

    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # sort candidate edges by score (lower is better)
    candidate_edges.sort()

    # for each candidate edge, connect two contours
    for _, cinfo_a, cinfo_b in candidate_edges:
        # if left and right are unassigned, join them
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # generate list of spans as output
    spans = []

    # until we have removed everything from the list
    while cinfo_list:

        # get the first on the list
        cinfo = cinfo_list[0]

        # Get the "first" one
        # keep following predecessors until none exists
        while cinfo.pred:
            cinfo = cinfo.pred

        # start a new span
        cur_span = []

        width = 0.0

        # follow successors til end of span
        while cinfo:
            # remove from list (sadly making this loop *also* O(n^2)
            cinfo_list.remove(cinfo)
            # add to span
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            # set successor
            cinfo = cinfo.succ

        # add if long enough, throw away if this span is too short
        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    if DEBUG_LEVEL >= 2:
        visualize_spans(name, small, pagemask, spans)

    return spans


def sample_spans(shape, spans):
    assert isinstance(shape, tuple)
    assert isinstance(spans, list)

    span_points = []

    for span in spans:

        contour_points = []

        for cinfo in span:

            '''
                Ex.
                    totals = 0 0 1 0 1 0 1 0 0      0
                             1 1 1 1 1 0 1 0 0   x  1
                             1 1 1 1 1 1 1 1 1      2

                           = 0 0 0 0 0 0 0 0 0
                             1 1 1 1 1 0 1 0 0
                             2 2 2 2 2 2 2 2 2
                        
                    means  = the mean over rows
            '''
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0) # sum over rows
            # means[i] will be the mean y coordinates of along x axis with value i
            means = totals / cinfo.mask.sum(axis=0)

            # Base coordinates
            xmin, ymin = cinfo.rect[:2]

            step = SPAN_PX_PER_STEP
            # Set start from the middle of the first window (step)
            start = int(((len(means)-1) % step) / 2)

            contour_points += [(x+xmin, means[x]+ymin)
                               for x in range(start, len(means), step)]

        contour_points = np.array(contour_points,
                                  dtype=np.float32).reshape((-1, 1, 2))

        contour_points = pix2norm(shape, contour_points)

        span_points.append(contour_points)

    return span_points


def keypoints_from_samples(name, small, pagemask, page_outline,
                           span_points):
    assert isinstance(name, str)
    assert isinstance(small, np.ndarray)
    assert isinstance(pagemask, np.ndarray) and small.shape[0:2] == pagemask.shape
    assert isinstance(page_outline, np.ndarray) and page_outline.shape == (4, 2)
    assert isinstance(span_points, list)

    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:

        # Compute the vector that can maximize the projection variance of the datapoints
        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)

        # The length of this spans (points[0] --> points[-1])
        weight = np.linalg.norm(points[-1] - points[0])

        all_evecs += evec * weight
        all_weights += weight

    # weighted sum of all PCA eigenvectors
    evec = all_evecs / all_weights

    x_dir = evec.flatten()

    # Normalize the x coordinate
    if x_dir[0] < 0:
        x_dir = -x_dir

    # The perpendicular vector of x_dir
    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline)
    # Normalize pixel coordinate to pagemask centered coordinates
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)

    px0 = px_coords.min() # the left most in the direction of x_dir
    px1 = px_coords.max() # the right most in the direction of y_dir

    py0 = py_coords.min()
    py1 = py_coords.max()

    '''
        In the x_dir, y_dir coordinate system (the author defined them as "page" coordinate),
        corners will look like this

                  |    
      (px0, py0)  |   (px1, py0)
        *---------|------*
        |         |      |
        ----------|----------> x_dir
        |         |      |
        |         |      |
        |         |      |
        *---------|------*
      (px0, py1)  |   (px1, py1)
                  v
                 y_dir
    '''

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir

    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []

    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        # Set (px0, py0) as the origin (from the perspective of the "page")
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    if DEBUG_LEVEL >= 2:
        visualize_span_points(name, small, span_points, corners)

    return corners, np.array(ycoords), xcoords


def visualize_contours(name, small, cinfo_list):

    regions = np.zeros_like(small)

    for j, cinfo in enumerate(cinfo_list):

        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)

    for j, cinfo in enumerate(cinfo_list):
        color = CCOLORS[j % len(CCOLORS)]
        color = tuple([c/4 for c in color])

        cv2.circle(display, fltp(cinfo.center), 3,
                   (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    debug_show(name, 1, 'contours', display)


def visualize_spans(name, small, pagemask, spans):

    regions = np.zeros_like(small)

    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        cv2.drawContours(regions, contours, -1,
                         CCOLORS[i*3 % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)
    display[pagemask == 0] /= 4

    debug_show(name, 2, 'spans', display)


def visualize_span_points(name, small, span_points, corners):

    display = small.copy()

    for i, points in enumerate(span_points):

        points = norm2pix(small.shape, points, False)

        mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
                                          None,
                                          maxComponents=1)

        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())

        point0 = mean + small_evec * (dps.min()-dpm)
        point1 = mean + small_evec * (dps.max()-dpm)

        for point in points:
            cv2.circle(display, fltp(point), 3,
                       CCOLORS[i % len(CCOLORS)], -1, cv2.LINE_AA)

        cv2.line(display, fltp(point0), fltp(point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.polylines(display, [norm2pix(small.shape, corners, True)],
                  True, (255, 255, 255))

    debug_show(name, 3, 'span points', display)


def imgsize(img):
    height, width = img.shape[:2]
    return '{}x{}'.format(width, height)


def make_keypoint_index(span_counts):
    '''
        Create indices that can be used to index 
    '''

    nspans = len(span_counts)
    npts = sum(span_counts)
    # keypoint_index[0]: is a dummy ( I personally think this is for original point fitting?? )
    keypoint_index = np.zeros((npts+1, 2), dtype=int)
    start = 1

    # Setting 8 is because params[0:8] contain rotation, translation and cubic slopes paramters
    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start+end, 1] = 8+i # [start:start+end] span points belong to which i-th span (or which ycoord) (8+i)
        start = end

    # each point's index (8 is for parameters offset, nspans is for ycoords offset)
    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans


    # The author use params[keypoint_index] to retrieve all x and y coordinates

    return keypoint_index


def optimize_params(name, small, dstpoints, span_counts, params):
    assert isinstance(small, np.ndarray)
    assert isinstance(dstpoints, np.ndarray) and dstpoints.shape[-1] == 2 # (x,y)
    assert isinstance(span_counts, list)

    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts)**2) # reprojection error

    print('  initial objective is', objective(params))

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, 'keypoints before', display)

    print('  optimizing', len(params), 'parameters...')
    start = datetime.datetime.now()
    # Adjust rotation, translation and cubic parameters so that the dstpoints can be fitted
    res = scipy.optimize.minimize(objective, params,
                                  method='Powell')
    end = datetime.datetime.now()
    print('  optimization took', round((end-start).total_seconds(), 2), 'sec.')
    print('  final objective is', res.fun)
    params = res.x

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 5, 'keypoints after', display)

    return params


def get_page_dims(corners, rough_dims, params):

    dst_br = corners[2].flatten()

    dims = np.array(rough_dims)

    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten())**2)

    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x

    print('  got page dims', dims[0], 'x', dims[1])

    return dims


def remap_image(name, img, small, page_dims, params):

    height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)

    width = round_nearest_multiple(height * page_dims[0] / page_dims[1],
                                   REMAP_DECIMATE)

    print('  output will be {}x{}'.format(width, height))

    height_small = height / REMAP_DECIMATE
    width_small = width / REMAP_DECIMATE

    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)

    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))

    page_xy_coords = page_xy_coords.astype(np.float32)

    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)

    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

    image_x_coords = cv2.resize(image_x_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)

    image_y_coords = cv2.resize(image_y_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    remapped = cv2.remap(img_gray, image_x_coords, image_y_coords,
                         cv2.INTER_CUBIC,
                         None, cv2.BORDER_REPLICATE)

    thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, ADAPTIVE_WINSZ, 25)

    pil_image = Image.fromarray(thresh)
    pil_image = pil_image.convert('1')

    threshfile = name + '_thresh.png'
    pil_image.save(threshfile, dpi=(OUTPUT_DPI, OUTPUT_DPI))

    if DEBUG_LEVEL >= 1:
        height = small.shape[0]
        width = int(round(height * float(thresh.shape[1])/thresh.shape[0]))
        display = cv2.resize(thresh, (width, height),
                             interpolation=cv2.INTER_AREA)
        debug_show(name, 6, 'output', display)

    return threshfile


def main():

    if len(sys.argv) < 2:
        print('usage:', sys.argv[0], 'IMAGE1 [IMAGE2 ...]')
        sys.exit(0)

    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != 'file':
        cv2.namedWindow(WINDOW_NAME)

    outfiles = []

    for imgfile in sys.argv[1:]:

        img = cv2.imread(imgfile)
        small = resize_to_screen(img)
        basename = os.path.basename(imgfile)
        name, _ = os.path.splitext(basename)

        print('loaded', basename, 'with size', imgsize(img),)
        print('and resized to', imgsize(small))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.0, 'original', small)

        pagemask, page_outline = get_page_extents(small)

        cinfo_list = get_contours(name, small, pagemask, 'text')
        spans = assemble_spans(name, small, pagemask, cinfo_list)

        if len(spans) < 3:
            print('  detecting lines because only', len(spans), 'text spans')
            cinfo_list = get_contours(name, small, pagemask, 'line')
            spans2 = assemble_spans(name, small, pagemask, cinfo_list)
            if len(spans2) > len(spans):
                spans = spans2

        if len(spans) < 1:
            print('skipping', name, 'because only', len(spans), 'spans')
            continue

        span_points = sample_spans(small.shape, spans)

        assert isinstance(span_points, list)

        print('  got', len(spans), 'spans')
        print('with', sum([len(pts) for pts in span_points]), 'points.')

        corners, ycoords, xcoords = keypoints_from_samples(name, small,
                                                           pagemask,
                                                           page_outline,
                                                           span_points)

        rough_dims, span_counts, params = get_default_params(corners,
                                                             ycoords, xcoords)

        dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
                              tuple(span_points))

        ###################### Optimization sections ######################
        params = optimize_params(name, small,
                                 dstpoints,
                                 span_counts, params)

        page_dims = get_page_dims(corners, rough_dims, params)
        ###################################################################

        outfile = remap_image(name, img, small, page_dims, params)

        outfiles.append(outfile)

        print('  wrote', outfile)
        print()

    print('to convert to PDF (requires ImageMagick):')
    print('  convert -compress Group4 ' + ' '.join(outfiles) + ' output.pdf')


if __name__ == '__main__':
    main()
