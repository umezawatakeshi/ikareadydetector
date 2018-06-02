# ikareadydetector
# Copyright (c) 2018 UMEZAWA Takeshi
# Licensed under GNU GPL version 2 or later

import cv2
from time import sleep

# 640x360 での「準備OK？」の領域
top=253
left=128
bottom=272
right=197

# ソース画像を比較関数に渡す前にかけるフィルタ
def pre_diff_filter(src):
	tmp = src
	tmp = tmp[top:bottom, left:right]
	tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	ret, tmp = cv2.threshold(tmp, 192, 255, cv2.THRESH_BINARY)
	return tmp

# ソース画像をウィンドウに表示する前にかけるフィルタ（デバッグ向け）
def pre_show_filter(src):
	tmp = src
	return tmp

# 画像のチャンネル数を取得する
# .shape がグレースケールだと 2-tuple、カラーだと 3-tuple を返すのを吸収する
def getch(src):
	h, w, *ch = src.shape
	if len(ch) == 0:
		return 1
	else:
		return ch[0]


tmpl = cv2.imread("template.png")
tmpl = cv2.resize(tmpl, (640, 360), cv2.INTER_CUBIC)
tmpl = pre_diff_filter(tmpl)

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("D:\\cap\\amarec(20180519-2234).avi")
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
	img = pre_diff_filter(captured)
	diff = cv2.absdiff(img, tmpl).sum() * 1.0 / (bottom-top) / (right-left) / getch(img)
	cv2.putText(captured, "diff=%6.2f" % diff, (0, 360), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
	detected = diff < 2.0
	if detected:
		cv2.putText(captured, "READY?", (320, 360), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

	# ウィンドウに描画
	cv2.imshow('Ika Private Match Ready Detector', pre_show_filter(captured))

	# q を押したら終了
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	sleep(1)

cap.release()
cv2.destroyAllWindows()
