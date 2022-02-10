- To face swap using webcam: 
	python real_time_face_swapping.py source.jpg
arg1: the image you would like to swap your own face with.

Required pkg:
	dlib
	scipy
	numpy
	cv2

- Pipeline: 
1. resize input image
2. detect faces in the resized input image using dlib's get_frontal_face_detector()
3. extract facial landmarks from detected face using dlib's off-the-shelf shape_predictor()
4. create convex hull from these facial landmarks using cv2.convexHull()
5. include landmark points off inner mouth and eyes regions into the convex hull. This way the mouth and eyes regions will follow the person's facial movement in the webcam.
6. create triangulation from convex hull using scipy's Delaunay()
7. start processing webcam video stream:
	* 7.1 resize each frame
	* 7.2 detect facial landmarks of each frame
	* 7.3 compare current landmarks with those of previous frame to estimate the "stablized" position of landmarks using Lucas-Kanade method
	* 7.4 the facial landmarks of the current frame will be 0.4*detected position+0.6*"stablized" position
	* 7.5 loop through each triangle. if the triangle is part of the eyes or mouth region, we skip. Otherwise, we warp the corresponding triangle from the input image onto current frame's triangular area.
	* 7.6 create binary mask of warped face and compute convex hull as well as center point of the warped face in the context of the current webcam frame.
	* 7.7 apply Poisson blending to correct the color tone using cv2.seeamlessClone()
	* 7.8 cv2.imshow() processed webcam frame
	
