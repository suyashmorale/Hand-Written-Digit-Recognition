from msilib.schema import Font
import matplotlib.pyplot as plt
import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2 
from tensorflow.python.keras.backend import constant
import os


pygame.init()

BOUNDARYINC = 10
WINDOWSIZEX = 640
WINDOWSIZEY = 480
WHITE = (255,255,255)
bg = pygame.image.load("board.jpg")


#INSIDE OF THE GAME LOOP
RED = (255,0,0)
IMAGESAVE = True
iswriting = False
FONT = pygame.font.Font('freesansbold.ttf', 32)
MODEL = load_model("model01.h5")
PREDICT = True
# Define the directory to save the images
SAVE_DIR = "digit_images"

LABELS = {0:'ZERO',1:'ONE',
          2:'TWO',3:'THREE',
          4:'FOUR',5:'FIVE',
          6:'SIX',7:'SEVEN',
          8:'EIGHT',9:'NINE'}

pygame.display.set_caption("DigitBoard")
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))
DISPLAYSURF.blit(bg, (0, 0))

number_xcord = []
number_ycord = []
img_cnt = 1

while(True):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 2, 10)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False

            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARYINC)
            rect_min_y, rect_max_y = max(0, number_ycord[0]-BOUNDARYINC ), min(number_ycord[-1]+BOUNDARYINC,WINDOWSIZEX)
            

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
            print(img_arr.shape)
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)

            # if IMAGESAVE:
            #     # img = cv2.resize(img_arr,(28,28))
            #     # img = np.pad(img, (10,10),'constant',constant_values=0)
            #     img = img_arr/255
            #     cv2.imwrite(os.path.join(SAVE_DIR, f"image_{img_cnt}.png"), img)
            #     img_cnt += 1

            if PREDICT:

                img = cv2.resize(img_arr,(28,28))
                img = np.pad(img, (10,10),'constant',constant_values=0)
                img = cv2.resize(img,(28,28))/255
                cv2.imwrite(os.path.join(SAVE_DIR, f"image_{img_cnt}.png"), img)
                label = str(LABELS[np.argmax(MODEL.predict(img.reshape(1,28,28,1)),axis=1)[0]])
                print(MODEL.predict(img.reshape(1,28,28,1)))
                textsurface = FONT.render(label, True, RED)
                textrecobj = textsurface.get_rect()
                textrecobj.left, textrecobj.bottom = rect_min_x, rect_max_y
                pygame.draw.rect(DISPLAYSURF,RED,pygame.Rect(rect_min_x,rect_min_y,rect_max_x-rect_min_x,rect_max_y-rect_min_y),2,1)
                DISPLAYSURF.blit(textsurface,textrecobj)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    DISPLAYSURF.fill(BLACK)


        pygame.display.update()