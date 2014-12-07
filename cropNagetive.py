#!/usr/bin/env python2

import numpy as np
import cv2
import sys

class App(object):
    def __init__(self):
        with open('l') as f:
            imagelist = [e.strip() for e in f]
        self.image = (cv2.imread(e, cv2.CV_LOAD_IMAGE_GRAYSCALE) for e in imagelist)
        self.index=0

        cv2.namedWindow('display')
        cv2.setMouseCallback('display', self.onmouse)

        self.tracking_state = None
        self.selection = None
        self.drag_start = None

        self.frame = self.image.next()

    def crop(self, selection):
        x0, y0, x1, y1 = selection
        #roi = self.frame[y0:y1, x0:x1]

        #for n in range(1, 8):
        for n,(w,h) in enumerate([(21,24), (28,32), (35,40), (49,56), (77,88), (133,152)]):
            for j in range(x0, x1, w):
                for i in range(y0, y1, h):
                    name="./nagetive/N_{1}_{0:05}.pgm".format(self.index, n)
                    sys.stdout.write(name+'\r')
                    roi = self.frame[i:i+h, j:j+w]
                    cv2.imwrite(name, cv2.resize(roi, (21, 24)))
                    self.index+=1

    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                self.selection = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.selection = (x0, y0, x1, y1)
            else:
                self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1

    def run(self):
        while 1:
            vis = self.frame.copy()
            if self.selection:
                x0, y0, x1, y1 = self.selection
                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)

            if self.tracking_state:
                self.tracking_state=0
                self.crop(self.selection)
                self.selection = None

            cv2.imshow('display', vis)
            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            elif ch == ord('a'):
                self.crop((0, 0, self.frame.shape[1], self.frame.shape[0]))
            elif ch == ord('n'):
                self.frame = self.image.next()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    App().run()
