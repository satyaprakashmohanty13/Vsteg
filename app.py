import streamlit as st
import os
import cv2
import numpy as np
import itertools
from PIL import Image
from pathlib import Path
import moviepy.editor
from os import path

quant = np.array([[16, 11, 10, 16, 24, 40, 51, 61],      # QUANTIZATION TABLE
                  [12, 12, 14, 19, 26, 58, 60, 55],    # required for DCT
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

class DCT():
    def __init__(self):  # Constructor
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0

    def encode_image(self, img, secret_msg):
        secret = secret_msg
        self.message = str(len(secret))+'*'+secret
        self.bitMess = self.toBits()

        row, col = img.shape[:2]
        self.oriRow, self.oriCol = row, col
        if((col/8)*(row/8) < len(secret)):
            st.error("Error: Message too large to encode in image")
            return False

        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img, row, col)

        row, col = img.shape[:2]

        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)

        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j, i) in itertools.product(range(0, row, 8),
                                                                                      range(0, col, 8))]

        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        quantizedDCT = [np.round(dct_Block/quant) for dct_Block in dctBlocks]

        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][6]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = int(self.bitMess[messIndex][letterIndex])
            DC = np.packbits(DC)
            DC = np.float32(DC)
            quantizedBlock[0][6] = DC
            letterIndex = letterIndex+1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break

        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        return sImg

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')
            bits.append(binval)
        return bits

    def addPadd(self, img, row, col):
        img = cv2.resize(img, (col+(8-col%8), row+(8-row%8)))
        return img

    def decode_image(self, img):
        row, col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0

        bImg, gImg, rImg = cv2.split(img)
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j, i) in itertools.product(range(0, row, 8),
                                                                            range(0, col, 8))]

        quantizedDCT = [img_Block/quant for img_Block in imgBlocks]
        i = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][6]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            buff += DC[7] << (7-i)
            i = 1+i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i = 0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize))+1:]
        return ''

def video_to_frames(input_video):
    if not os.path.exists("video"):
        os.makedirs("video")
    with open(os.path.join("video", input_video.name), "wb") as f:
        f.write(input_video.getbuffer())

    cap = cv2.VideoCapture(os.path.join("video", input_video.name))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not os.path.exists("frames"):
        os.makedirs("frames")

    path_to_save = './frames'
    current_frame = 0

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        name = 'frame' + str(current_frame) + '.png'
        cv2.imwrite(path.join(path_to_save, name), frame)
        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    video = moviepy.editor.VideoFileClip(os.path.join("video", input_video.name))
    audio = video.audio
    if audio:
        audio.write_audiofile('./video/output.mp3')

def frames_to_video():
    path = "./frames"
    frames = []
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    files.sort(key=lambda x: int(x[5:-4]))

    for f in files:
        filename = os.path.join(path, f)
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        height, width, channels = img.shape
        size = (width, height)
        frames.append(img)

    fps = 30
    output = cv2.VideoWriter('./video/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for img in frames:
        output.write(img)
    output.release()

    input_video = "./video/output.mp4"
    input_audio = "./video/output.mp3"
    output_video = "./video/final.mp4"

    videoClip = moviepy.editor.VideoFileClip(input_video)
    if os.path.exists(input_audio):
        audioClip = moviepy.editor.AudioFileClip(input_audio)
        finalClip = videoClip.set_audio(audioClip)
        finalClip.write_videofile(output_video, fps=fps)
    else:
        videoClip.write_videofile(output_video, fps=fps)

def cleanup():
    if os.path.exists("frames"):
        for f in os.listdir("frames"):
            os.remove(os.path.join("frames", f))
        os.rmdir("frames")
    if os.path.exists("video"):
        for f in os.listdir("video"):
            os.remove(os.path.join("video", f))
        os.rmdir("video")

st.title("Video Steganography")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Encode", "Decode"])

if app_mode == "Encode":
    st.header("Encode Video")
    input_video = st.file_uploader("Upload Video", type=["mp4"])
    secret_msg = st.text_area("Enter Secret Message")
    if st.button("Encode"):
        if input_video is not None and secret_msg is not None:
            video_to_frames(input_video)
            frames_dir = "frames"
            files = [f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]
            img_name = files[len(files)//2]
            img_path = os.path.join(frames_dir, img_name)
            if not os.path.isfile(img_path):
                st.error("Image not found")
            else:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                encoded_img = DCT().encode_image(img, secret_msg)
                cv2.imwrite(img_path, encoded_img)
                frames_to_video()
                st.success("Encoding Successful!")
                st.video("video/final.mp4")
                cleanup()

elif app_mode == "Decode":
    st.header("Decode Video")
    output_video = st.file_uploader("Upload Video", type=["mp4"])
    if st.button("Decode"):
        if output_video is not None:
            video_to_frames(output_video)
            path = "./frames/"
            files = [f for f in os.listdir(path) if not f.startswith('.')]
            for file in files:
                filename = os.path.join(path, file)
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                hidden_text = DCT().decode_image(img)
                if hidden_text != "":
                    st.success(f"Secret Message: {hidden_text}")
                    cleanup()
                    break
