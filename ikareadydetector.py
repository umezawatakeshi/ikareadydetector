# ikareadydetector
# Copyright (c) 2018 UMEZAWA Takeshi
# Licensed under GNU GPL version 2 or later

import cv2
from time import sleep

class Matcher:
	@staticmethod
	def pre_filter(src):
		tmp = src
		tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
		_, tmp = cv2.threshold(tmp, 192, 255, cv2.THRESH_BINARY)
		return tmp

	def __init__(self, filename, top, left, bottom, right):
		tmp = cv2.imread(filename)
		tmp = cv2.resize(tmp, (640, 360), cv2.INTER_CUBIC)
		tmp = Matcher.pre_filter(tmp)
		tmp = tmp[top:bottom, left:right]
		self.img = tmp
		self.top = top
		self.left = left
		self.bottom = bottom
		self.right = right
		self.pixels = (bottom - top) * (right - left)

	def match(self, src):
		tgt = src[self.top:self.bottom, self.left:self.right]
		diff = cv2.absdiff(tgt, self.img).sum() * 1.0 / self.pixels
		return (diff < 2.0, diff)


tmpl_ready = Matcher("templates/ready.png", 253, 128, 272, 197)

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("amarec(20180519-2234).avi")
#cap = cv2.VideoCapture("test2.avi")

while True:
	ret, captured = cap.read()
	if not ret:
		print("cannot read from capture")
		break

	# 入力画像を 16:9 にクロップしてから 640x360 にリサイズする
	caph, capw, *_ = captured.shape
	if capw * 9 < caph * 16:
		bandh = (caph - capw * 9 // 16) // 2;
		captured = captured[bandh:caph-bandh]
		caph -= bandh * 2
	elif capw * 9 > caph * 16:
		bandw = (capw - caph * 16 // 9) // 2;
		captured = captured[0:caph, bandw:capw-bandw]
		capw -= bandw * 2
	if capw != 640:
		captured = cv2.resize(captured, (640, 360), cv2.INTER_CUBIC)

	# テンプレート画像との差異を計算して検出
	# 検出結果は画像下部に描画
	img = Matcher.pre_filter(captured)
	detected_ready, diff_ready = tmpl_ready.match(img)
	cv2.putText(captured, "diff=%6.2f" % diff_ready, (0, 360), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
	if detected_ready:
		cv2.putText(captured, "READY?", (320, 360), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

	# ウィンドウに描画
	cv2.imshow('Ika Private Match Ready Detector', captured)

	# q を押したら終了
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	sleep(1)

cap.release()
cv2.destroyAllWindows()
