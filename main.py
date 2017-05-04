import cv2
import pyautogui as pag

import numpy as np
from PIL import ImageGrab


def start_game(difficulty):
    difficulty_levels = {"beginner": (470, 400), "advanced": (470, 450), "expert": (470, 495)}
    difficulty_pos = difficulty_levels[difficulty]
    pag.moveTo(difficulty_pos[0], difficulty_pos[1])
    pag.click()

def reload():
    button_pos = (290, 540)
    pag.moveTo(button_pos[0], button_pos[1])
    pag.click()


def main():
    mog2 = cv2.createBackgroundSubtractorMOG2()
    ammo = 6
    start_game("beginner")
    while True:
        screen = np.array(ImageGrab.grab((0, 0, 800, 640)))
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        mask = mog2.apply(screen_gray)
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000 or area > 5000:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(screen, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Shoot in the center of target for maximum score
            target_x = x + w/2
            target_y = y + h/2
            pag.moveTo(target_x, target_y)
            pag.click()
            ammo -= 1
            if ammo == 0:
                reload()
                ammo = 6
        cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('mask', mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

main()
