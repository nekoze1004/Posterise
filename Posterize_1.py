'''
Created on 2017/07/23

@author: nekoze1004
'''
import cv2
import numpy as np
import sys
import time
from numba import jit


# 引数の文字列はYesか
def IsYes(yn):
    if ((yn == "y") | (yn == "Y") | (yn == "yes") | (yn == "Yes") | (yn == "YES")):
        return True
    else:
        return False


# 引数の文字列はNoか
def IsNo(yn):
    if ((yn == "n") | (yn == "N") | (yn == "no") | (yn == "No") | (yn == "NO")):
        return True
    else:
        return False


# 引数はYesでもNoでもないか
def IsNotYN(yn):
    if (IsYes(yn) | IsNo(yn)):
        return False
    else:
        return True


# 二値画像の色を反転させる
@jit
def reverse(binaryImg):
    copyImg = np.copy(binaryImg)
    for i in range(binaryImg.shape[0]):
        for j in range(binaryImg.shape[1]):
            if binaryImg[i, j] == 0:
                copyImg[i, j] = 255
            else:
                copyImg[i, j] = 0
    return copyImg


@jit
def posterize(img, pos):
    r = np.zeros(img.shape, np.uint8)
    if pos == 1:
        # もし１が入ってきたら、普通の処理では０で割る動作があるので例外的に処理
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] <= 127:
                    r[i, j] = 0
                else:
                    r[i, j] = 255
    elif pos >= 256:
        # もし256以上が入ってきたら、画像をそっくりそのまま返す
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                r[i, j] = img[i, j]
    else:
        # 2~255までを想定
        # 読み込んだ元画像の全画素を巡回して、閾値と比べて当てはまる階級の値を
        # 書き出し先の同じ場所の画素に入れていく
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for p in range(0, pos):
                    # pos=3のとき、0-85,86-170,171-255で分けられる
                    # それぞれ0,127,255が入れられる
                    if (img[i, j] >= ((255 // pos) * p) + 1) & (img[i, j] <= (255 // pos) * (p + 1)):
                        r[i, j] = (255 // (pos - 1)) * p
                        if p + 1 == pos:
                            # 255//posが本来２５５ぴったしになるといいが、現実は非常なので、２５５になるべき者たちが２５５になれるようにしている（謎の言い回し）
                            if (img[i, j] >= (255 // pos) * (p + 1)) & (img[i, j] <= 255):
                                r[i, j] = 255
    return r


@jit
def masked(base, mask):
    r = np.copy(base)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                r[i, j] = 0
    return r


if __name__ == "__main__":

    print("ポスタライズします\n")
    print("読み込む画像名 NewImageフォルダ内")
    inputImg = input(">>> ")

    img = cv2.imread("NewImages/" + inputImg + ".png")
    if img is None:
        img = cv2.imread("NewImages/" + inputImg + ".jpg")
        if img is None:
            print("認識できません。")
            sys.exit()

    print("モノクロにしますか？(y/n)")
    mono = input(">>> ")
    while IsNotYN(mono):
        print("yかnを入力してください。")
        print("モノクロにしますか？(y/n)")
        mono = input(">>> ")

    if IsYes(mono):
        print("モノクロでポスタライズします。")
        # imgをグレースケール化
        GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 画像下処理　ぼかし
        GaussGrayImg = cv2.GaussianBlur(GrayImg, (5, 5), 0)

        # Canny法を用いて輪郭を検出する
        sita = 50
        ue = 100
        CannyGaussGrayImg = cv2.Canny(GaussGrayImg, int(sita), int(ue))

        # Canny法で求めた線の白黒を反転させる
        ReverseCannyGaussGrayImg = reverse(CannyGaussGrayImg)

        sen = ReverseCannyGaussGrayImg

        print("線を太くしますか？ (y/n)")
        hutoi = input(">>> ")
        while IsNotYN(hutoi):
            print("yかnを入力してください。")
            print("線を太くしますか？(y/n)")
            mono = input(">>> ")

        if IsYes(hutoi):
            near8 = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             np.uint8)
            ErosionReverseCannyGaussGrayImg = cv2.erode(ReverseCannyGaussGrayImg, near8, iterations=1)
            sen = ErosionReverseCannyGaussGrayImg

        print("ポスタライズ階調を入力 n<10推奨")
        p = input(">>> ")
        pos = int(p)

        start = time.time()

        PosterizeImg = posterize(GaussGrayImg, pos)

        endTime = time.time() - start
        print("Posterize time:" + format(endTime) + "[sec]")

        result = masked(PosterizeImg, sen)

        # 名は体を表すファイル名をつける
        fName = inputImg + "GrayPosterize.png"
        cv2.imshow("results " + fName, result)  # ポスタライズの結果を表示する
        cv2.imwrite(fName, result)  # ポスタライズの結果を保存する
        cv2.waitKey(0)  # なにかキーを押すと↓
        cv2.destroyAllWindows()  # 表示ウインドウが閉じる
    else:
        # 画像下処理　ぼかし
        GaussImg = cv2.GaussianBlur(img, (5, 5), 0)

        # Canny法を用いて輪郭を検出する
        sita = 50
        ue = 100
        CannyGaussImg = cv2.Canny(GaussImg, int(sita), int(ue))

        # Canny法で求めた線の白黒を反転させる
        ReverseCannyGaussImg = reverse(CannyGaussImg)

        sen = ReverseCannyGaussImg

        print("線を太くしますか？ (y/n)")
        hutoi = input(">>> ")
        while IsNotYN(hutoi):
            print("yかnを入力してください。")
            print("線を太くしますか？(y/n)")
            mono = input(">>> ")

        if IsYes(hutoi):
            near8 = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             np.uint8)
            ErosionReverseCannyGaussGrayImg = cv2.erode(ReverseCannyGaussImg, near8, iterations=1)
            sen = ErosionReverseCannyGaussGrayImg

        print("ポスタライズ階調を入力 n<10推奨")
        p = input(">>> ")
        pos = int(p)

        start = time.time()

        imgB = img[:, :, 0]
        imgG = img[:, :, 1]
        imgR = img[:, :, 2]

        PosterizeImgB = posterize(imgB, pos)
        PosterizeImgG = posterize(imgG, pos)
        PosterizeImgR = posterize(imgR, pos)

        PosterizeImg = np.copy(img)
        PosterizeImg[:, :, 0] = PosterizeImgB[:, :]
        PosterizeImg[:, :, 1] = PosterizeImgG[:, :]
        PosterizeImg[:, :, 2] = PosterizeImgR[:, :]
        endTime = time.time() - start
        print("Posterize time:" + format(endTime) + "[sec]")

        result = masked(PosterizeImg, sen)

        # 名は体を表すファイル名をつける
        fName = inputImg + "Posterize.png"
        cv2.imshow("results " + fName, result)  # ポスタライズの結果を表示する
        cv2.imwrite(fName, result)  # ポスタライズの結果を保存する
        cv2.waitKey(0)  # なにかキーを押すと↓
        cv2.destroyAllWindows()  # 表示ウインドウが閉じる