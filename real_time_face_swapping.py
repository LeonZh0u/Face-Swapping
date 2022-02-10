import math
import sys
from scipy.spatial import Delaunay
import cv2
import dlib
import numpy as np


WEBCAM_HEIGHT = 480
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def resize_image(img, ratio):
    small_img = cv2.resize(
        img,
        None,
        fx=1.0 / ratio,
        fy=1.0 / ratio,
        interpolation=cv2.INTER_LINEAR,
    )
    return small_img

def detect_facial_landmarks(img):
    small_img=resize_image(img,1.5)

    # use the biggest face
    rect = max(detector(small_img), key=lambda rect: rect.area())

    scaled_rect = dlib.rectangle(
        int(rect.left() * 1.5),
        int(rect.top() * 1.5),
        int(rect.right() * 1.5),
        int(rect.bottom() * 1.5),
    )
    landmarks = predictor(img, scaled_rect)

    return [(point.x, point.y) for point in landmarks.parts()]

def warp_triangle(img1, img2, p1, p2):
    rect1 = cv2.boundingRect(np.float32([p1]))

    img1_cropped = img1[rect1[1] : rect1[1] + rect1[3], rect1[0] : rect1[0] + rect1[2]]

    rect2 = cv2.boundingRect(np.float32([p2]))
    #account for the offset when switching coordinates to cropped image
    p1 = [
        ((p1[0][0] - rect1[0]), (p1[0][1] - rect1[1])),
        ((p1[1][0] - rect1[0]), (p1[1][1] - rect1[1])),
        ((p1[2][0] - rect1[0]), (p1[2][1] - rect1[1])),
    ]
    p2 = [
        ((p2[0][0] - rect2[0]), (p2[0][1] - rect2[1])),
        ((p2[1][0] - rect2[0]), (p2[1][1] - rect2[1])),
        ((p2[2][0] - rect2[0]), (p2[2][1] - rect2[1])),
    ]
    #create binary mask: 1 inside triangle, 0 elsewhere
    mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(p2), (1.0, 1.0, 1.0), 16,0)

    #calculate Affine transformation matrix
    mat = cv2.getAffineTransform(np.float32(p1), np.float32(p2))
    #apply matrix to warp the triangle
    img2_cropped = cv2.warpAffine(
        img1_cropped,
        mat,
        (rect2[2], rect2[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    #only keep the image inside the triangle for the warped rectangular cropped image
    img2_cropped = img2_cropped * mask
    img2_cropped_slice = np.index_exp[
        rect2[1] : rect2[1] + rect2[3], rect2[0] : rect2[0] + rect2[2]
    ]
    #add the warped image onto the webcam frame
    img2[img2_cropped_slice] = img2[img2_cropped_slice] * ((1.0, 1.0, 1.0) - mask)
    img2[img2_cropped_slice] = img2[img2_cropped_slice] + img2_cropped

if __name__ == '__main__':
    img1 = cv2.imread(sys.argv[1])
    height, width = img1.shape[:2]
    img1 = resize_image(img1, np.float32(height) / WEBCAM_HEIGHT)

    points1 = detect_facial_landmarks(img1)
    #manually widen the eyes region to allow for more eye movement.
    x,y=points1[40]
    points1[40]=(x,y+2)
    x,y=points1[37]
    points1[37]=(x,y-2)

    x,y=points1[39]
    points1[39]=(x-2,y)
    x,y=points1[42]
    points1[42]=(x+2,y)

    x,y=points1[46]
    points1[46]=(x,y+2)
    x,y=points1[43]
    points1[43]=(x,y-2)

    x,y=points1[47]
    points1[47]=(x,y+2)
    x,y=points1[44]
    points1[44]=(x,y-2)

    #obtain points of convex hull, including mouth and eyes points
    original_hull_index = cv2.convexHull(np.array(points1), returnPoints=False)
    mouth_points = [[60],[61],[62],[63],[64],[65],[66],[67]]
    eyes_points =[[37],[39],[40],[42],[43],[44],[46],[47]]

    hull1 = []
    hull_index = np.concatenate((original_hull_index, mouth_points, eyes_points))

    # Helper map
    to_index = {i:elem[0] for i, elem in enumerate(hull_index)}
    hull1 = [points1[hull_index_element[0]] for hull_index_element in hull_index]

    #triangulate the points
    delaunay_triangles = Delaunay(hull1).simplices

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width, frame_height) 
    sigma = 100
    first_frame = False
    #start processing webcam video
    while(True):
        ret, img2 = cap.read()
        if not ret:
            continue

        height, width = img2.shape[:2]
        img2 = resize_image(img2, np.float32(height) / WEBCAM_HEIGHT)
        try:
            points2 = detect_facial_landmarks(img2)
            x,y=points2[40]
            points2[40]=(x,y+2)
            x,y=points2[37]
            points2[37]=(x,y-2)

            x,y=points2[41]
            points2[41]=(x,y+2)
            x,y=points2[38]
            points2[38]=(x,y-2)

            x,y=points2[46]
            points2[46]=(x,y+2)
            x,y=points2[43]
            points2[43]=(x,y-2)

            x,y=points2[47]
            points2[47]=(x,y+2)
            x,y=points2[44]
            points2[44]=(x,y-2)

        except Exception:
            pass
        else:
            #creat convex hull for current frame
            hull2 = [points2[hull_index_element[0]] for hull_index_element in hull_index]
            original_hull2 = [
                points2[hull_index_element[0]] for hull_index_element in original_hull_index
            ]

            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            #record convex hull of previous frame for optical flow stabilization
            if first_frame is False:
                hull2_prev = np.array(hull2, np.float32)
                img2_gray_prev = np.copy(img2_gray)
                first_frame = True
            else:
                hull2_next, *_ = cv2.calcOpticalFlowPyrLK(
                    img2_gray_prev,
                    img2_gray,
                    hull2_prev,
                    np.array(hull2, np.float32),
                    winSize=(101, 101),
                    maxLevel=5,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001),
                )

                for i, _ in enumerate(hull2):
                    hull2[i] = 0.4 * np.array(hull2[i]) + 0.6 * hull2_next[i]

                hull2_prev = np.array(hull2, np.float32)
                img2_gray_prev = img2_gray

            #placeholder for warped input image
            img1_warped = np.copy(img2)
            img1_warped = np.float32(img1_warped)

            for tri in delaunay_triangles:
                #skip eyes and mouth region from warping
                mouth_points_set = set(mp[0] for mp in mouth_points)
                eyes_points_set = set(ep[0] for ep in eyes_points)
                if (
                    (to_index[tri[0]] in mouth_points_set
                    and to_index[tri[1]] in mouth_points_set
                    and to_index[tri[2]] in mouth_points_set)
                    or(to_index[tri[0]] in eyes_points_set
                    and to_index[tri[1]] in eyes_points_set
                    and to_index[tri[2]] in eyes_points_set)
                ):
                    continue
                #create triples of coordinates for both inout image and current frame to create warped triangles
                p1 = [points1[to_index[tri[0]]], points1[to_index[tri[1]]], points1[to_index[tri[2]]]]
                p2 = [hull2[tri[0]],hull2[tri[1]],hull2[tri[2]]]

                warp_triangle(img1, img1_warped, p1, p2)

            #create binary mask and bounding rectangle as well as the center point of warped face for seamlessClone()
            mask = np.zeros(img2.shape, dtype=img2.dtype)
            cv2.fillConvexPoly(mask, np.int32(original_hull2), (255, 255, 255))
            rect = cv2.boundingRect(np.float32([original_hull2]))
            center = (rect[0] + int(rect[2] / 2), rect[1] + int(rect[3] / 2))
            #perform Poisson color blending
            img2 = cv2.seamlessClone(np.uint8(img1_warped), img2, mask, center, cv2.NORMAL_CLONE)
            
        cv2.imshow("camera", img2)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
