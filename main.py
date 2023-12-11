import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Canvas, ttk
from PIL import Image, ImageTk
import HandTrackingModule as htm
import math
from sklearn.neighbors import KNeighborsClassifier

import speech_recognition as sr
import pyttsx3
import threading
import pickle


facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/name.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print(LABELS)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Face detection

detector = htm.handDetector(detectionCon=1)

tipIds = [ 4, 8, 12, 16 ,20]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Create a Tkinter window
root = tk.Tk()
root.title("Hand Tracking with Skeleton and Bounding Box")

# Create two side-by-side frames
frame_main = ttk.Frame(root)
frame_hand = ttk.Frame(root)
frame_main.pack(side="left", padx=10, pady=10)
frame_hand.pack(side="right", padx=10, pady=10)


# Create canvases for the main frame and hand view
canvas_main = Canvas(frame_main, width=640, height=480)
canvas_main.pack()
canvas_hand = Canvas(frame_hand, width=300, height=300)
canvas_hand.pack()

#State of motor
State_motor = ttk.Label(root, text = "Fact of the Day", font =("Courier", 14) )
State_motor.pack()

#Function for voice 


# Function to handle speech recognitionv
text = ''
def record_speech():
    global text
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Recording... Speak something!")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        text_display.config(text="Recognized Text: " + text)
    except sr.UnknownValueError:
        text_display.config(text="Speech not recognized")
        text = ''
    except sr.RequestError:
        text_display.config(text="Could not request results; check your internet connection")

# Function to prompt the user to speak
def say_something():
    engine = pyttsx3.init()
    engine.say("Say something, please")
    engine.runAndWait()

# Function to start speech recognition in a separate thread
def record_and_recognize():
    threading.Thread(target=say_something).start()
    threading.Thread(target=record_speech).start()

# Create a button to trigger speech recognition
button = tk.Button(root, text="Record and Recognize", command=record_and_recognize)
button.pack(pady=20)

# Create a label to display recognized text
text_display = tk.Label(root, text="Recognized Text Will Appear Here")
text_display.pack()


#Find which ngon tay gio len
def find_rise(lmList):
    fingers = []
    if len(lmList) != 0 :
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            #compare vi tri  tren khop tay
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                #if finger open
                fingers.append(1)
            else:
                fingers.append(0)
    return fingers

def is_touch(coor_finger, rect):
    w = 100
    h = 40
    convert_dict = {0 : 'Red', 1: 'Green', 2: 'Blue'}
    for i in range(len(rect)):
        if ( (rect[i][0] + w > coor_finger[0] > rect[i][0]) and ( rect[i][1] + h > coor_finger[1] > rect[i][1])):
            return convert_dict[i]

contac = []
output = []
def update_frames():
    global contac, output
    output = [0]
    ret, frame = cap.read()

    #frame = cv2.flip(frame, 1) 
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        crop_img = crop_img/255 #preprocess
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)

        distances, indices = knn.kneighbors(resized_img)
        min_distance = distances[0][0]
        #print(min_distance)

        if min_distance > 12:
            output[0] = 'Unknown'

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0] ) + ':'+str (min_distance) , (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if (output[0]!= 0 and output[0]!= 'Unknown' ):
        #print(output)
        try:
            State_motor.config(text= (text))
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                
                for landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate and draw the bounding box
                    landmarks_points = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in landmarks.landmark]
                    
                    #Get coordinate of each point of hand
                    lmList = []
                    myHand = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myHand.landmark):
                        #id is index of landmark : you can find it in mediapipe hand detection 
                        # print(id, lm)
                        h, w, c = frame.shape
                        #coordinate of landmark (finger)
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # print(id, cx, cy)
                        #save coordinate and index to a list lm
                        lmList.append([id, cx, cy])

                    fingers = find_rise(lmList)
                    #cái này hơi khác ngón cái cụp lại = 1, mở ra lại = 0, các ngón khác thì mở ra là 1 còn đóng lại là 1
                    
                    #Nếu mở 2 ngón tay cái và giữa ra thì sẽ bắt đầu tốc độ
                    if fingers == [0,1,0,0,0]:
                        cv2.circle(frame,(lmList[8][1],lmList[8][2],),10,(255,255,0))
                        cv2.circle(frame,(lmList[4][1],lmList[4][2],),10,(255,255,0))
                        cv2.line(frame, (lmList[8][1],lmList[8][2],),(lmList[4][1],lmList[4][2],),(45,255,200))
                        distance = int(math.sqrt((lmList[8][1] - lmList[8][2])**2 - (lmList[4][1] - lmList[4][2])**2))
                        #distance = int(math.hypot(lmList[8][1] - lmList[8][2], lmList[4][1] - lmList[4][2]))
                        if distance >= 200:
                                distance = 255
                        elif distance <=30:
                                distance = 0
                            
                        print(distance)
                        
                        contac.append(0)
                        
                    # Voice - recognition : Gio 4 ngon
                    if fingers == [1,0,1,1,1]:

                        #Cai nay de luu lich su x
                        contac.append(1)
                        #print(contac)
                        print('Voice')
                        if (len(contac) >= 2):
                            #Neu là trước nó là 0 tức là trước nó có cử chỉ khác thì mới bật voice nếu mà trước nó đã là 1 rồi thì thôi (là lúc đó bị lặp)
                            if (contac[-1] == 1 and contac[-2] == 0):
                                button.invoke()
                                contac = []
                    else:
                        contac.append(0)
                        
                    x, y, w, h = cv2.boundingRect(np.array(landmarks_points))
                    first_point = (x - 10, y - 10)
                    last_point = (x + w + 10, y + h + 10)
                    cv2.rectangle(frame, first_point, last_point, (0, 255, 0), 2)

                    # Display bounding box size in the main frame
                    cv2.putText(frame, f'Width: {w}, Height: {h}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Create a smaller window to display the hand and bounding box size
                    hand_frame = frame[y - 10:y + h + 10, x - 10:x + w + 10]

                    # Convert hand_frame to PIL format for display in the Tkinter canvas
                    hand_image = Image.fromarray(cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB))
                    hand_photo = ImageTk.PhotoImage(image=hand_image)
                    canvas_hand.create_image(0, 0, image=hand_photo, anchor=tk.NW)
                    canvas_hand.hand_photo = hand_photo
            else:
                canvas_hand.delete("all")  # Clear the hand view canvas if hand is not detected
        except Exception as e:
            print(f"An error occurred: {e}")

    # Convert the main frame to PIL format for display in the Tkinter canvas
    main_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    main_photo = ImageTk.PhotoImage(image=main_image)
    canvas_main.create_image(0, 0, image=main_photo, anchor=tk.NW)
    canvas_main.main_photo = main_photo

    # Schedule the update after a delay (you can adjust this delay)
    root.after(10, update_frames)

# Start the frame update loop
update_frames()

# Function to close the application
def close_app():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# Create a button to close the application
close_button = ttk.Button(root, text="Close", command=close_app)
close_button.pack()

root.mainloop()
