import pyautogui
import time
import os

pyautogui.PAUSE = 0.3


def open_chrome():
    pyautogui.press("win")
    time.sleep(1)
    pyautogui.write("chrome")
    pyautogui.press("enter")


def open_notepad():
    pyautogui.press("win")
    time.sleep(1)
    pyautogui.write("notepad")
    pyautogui.press("enter")


def scroll_down():
    pyautogui.scroll(-500)


def scroll_up():
    pyautogui.scroll(500)


def click():
    pyautogui.click()


def stop_system():
    print("🛑 Stopping assistant")
    os._exit(0)
