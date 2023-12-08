import mediapipe as mp
import cv2
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
    

class HandDetector:

    
    def __init__(self, maxhands=2,min_dtct_cnfdnc=0.5,min_trckn_cnfdnc=0.5):
       
       self.hands = mp_hands.Hands(max_num_hands = maxhands,min_detection_confidence = min_dtct_cnfdnc, min_tracking_confidence= min_trckn_cnfdnc)
    
    def findHandLMs(self, img, hands=0, draw=False):
        originalImage = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img)
        land_mark_list = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[hands]

            for hid, land_mark in enumerate(hand.landmark):
                imgH, imgW, imgC = originalImage.shape
                posX, posY = int(land_mark.x * imgW), int(land_mark.y * imgH)
                land_mark_list.append([hid, posX, posY])
            
            if draw:
                mp_draw.draw_landmarks(originalImage, hand, None)
        return land_mark_list

