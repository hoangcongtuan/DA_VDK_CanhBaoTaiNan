from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import imutils
import playsound
import numpy as numpy
import time
import dlib
import cv2

EYE_AR_THREASHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 1

COUNTER = 0
TOTAL = 0

alarmOn = False

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)
	return ear
def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[0], mouth[2])
	B = dist.euclidean(mouth[1], mouth[3])

	mar = B * 1.0 / A
	return mar

def sound_alarm(path):
	playsound.playsound(path)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dat/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(noseStart, noseEnd) = face_utils.FACIAL_LANDMARKS_IDXS['nose']
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

vs = cv2.VideoCapture(0)
startSleep = False;

time.sleep(1.0)

while True:
	ret, frame = vs.read()
	frame = imutils.resize(frame, width = 600)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart : lEnd]
		rightEye = shape[rStart : rEnd]
		nose = shape[noseStart: noseEnd]
		mouth =shape[mouthStart : mouthEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		#calculate mouth ratio
		mouthLeft = mouth[0]
		mouthTop = mouth[3]
		mouthRight = mouth[6]
		mouthBottom = mouth[9]
		moutSimple = (mouthLeft, mouthTop, mouthRight, mouthBottom)

		ear = (leftEAR + rightEAR) / 2.0

		mar = mouth_aspect_ratio(moutSimple)

		#draw
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		noseHull = cv2.convexHull(nose)
		#mouthHull = cv2.convexHull(mouthLeft, mouthTop, mouthRight, mouthBottom)
		#mouthHull = cv2.convexHull(mouth)





		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [noseHull], -1, (255, 0, 0), 1)
		#cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
		cv2.line(frame, (mouthLeft[0], mouthLeft[1]), (mouthRight[0], mouthRight[1]), (0, 255, 0), 1)
		cv2.line(frame, (mouthTop[0], mouthTop[1]), (mouthBottom[0], mouthBottom[1]), (0, 255, 0), 1)


		if (ear < EYE_AR_THREASHOLD):
			if not startSleep:
				startSleep = True
				startTime = cv2.getTickCount()
			endTime = cv2.getTickCount()
			if (endTime - startTime >= 2 * cv2.getTickFrequency()):
				#drowsiness
				if not alarmOn:
					alarmOn = True
					print("Alarm on\n")
					t = Thread(target=sound_alarm,
							args=('audio/alarm.wav',))
					t.deamon = True
					t.start()
				
			COUNTER += 1
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			startSleep = False
			alarmOn = False
			COUNTER = 0
		#Write text

		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "Left EAR: {:.2f}".format(leftEAR), (300, 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		cv2.putText(frame, "Right EAR: {:.2f}".format(rightEAR), (300, 80),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 110),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	fps = vs.get(cv2.CAP_PROP_FPS)
	cv2.putText(frame, "FPS: {}".format(fps), (10, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break;
cv2.destroyAllWindows()
vs.release()


