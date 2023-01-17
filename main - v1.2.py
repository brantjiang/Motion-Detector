"""
Brant Jiang

General testing 
"""
import cv2
from Media import Photo, Video

def main(): 
    vid=Video("bird.mp4", 0.5)
    print(vid.findMotion())

if __name__ == "__main__":
    main()
