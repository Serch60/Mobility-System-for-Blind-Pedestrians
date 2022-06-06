#pip3 install pydub
#pip3 install playsound

import cv2
import numpy as np
from playsound import playsound
from time import time

import speech_recognition as sr

print('The System is ON')
playsound('/home/sergio/Escritorio/Mobility_System/Audio/system_on.mp3')
r = sr.Recognizer() 
while True:
    with sr.Microphone() as source:
        audio = r.listen(source)
 
        try:
            text = r.recognize_google(audio)
            print('You said: {}'.format(text))
            print(text)
            if "crosswalk" in text:
                playsound('/home/sergio/Escritorio/Mobility_System/Audio/said_crosswalk.mp3')

                cont=1
                while (cont<=10):
                    tiempo_inicial = time()
                    cap = cv2.VideoCapture(0)

                    leido, frame = cap.read()
                    if leido == True:
                        cv2.imwrite("crosswalk.png", frame)
                        print("Captured photo")
                    else:
                        print("Error accessing camera")

                    cap.release()


                    # Load Yolo
                    net = cv2.dnn.readNet("yolov3_training_last_4.weights", "yolov3_testing.cfg")
                    classes = ["cruce"]
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                    colors = np.random.uniform(0, 255, size=(len(classes), 3))

                    # Loading image
                    #img = cv2.imread("foto.png")
                    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
                    height, width, channels = img.shape

                    # Detecting objects
                    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    # Showing informations on the screen
                    class_ids = []
                    confidences = []
                    boxes = []
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                # Object detected
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                # Rectangle coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    #print(indexes)
                    
                    font = cv2.FONT_HERSHEY_PLAIN
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            print (label)
                            playsound('/home/sergio/Escritorio/Mobility_System/Audio/crosswalk_presence.mp3')
                            color = colors[i]
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


                    #cv2.imshow("Output", img)
                    #cv2.imwrite("cruce2.jpg", img)
                    tiempo_final = time()
                    tiempo_ejecucion = tiempo_final - tiempo_inicial
                    print("Execution time ",tiempo_ejecucion)
                    
                    #cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cont+=1

            if "traffic" in text:
                playsound('/home/sergio/Escritorio/Mobility_System/Audio/said_traffic_light.mp3')

                cont=1
                while (cont<=20):
                    tiempo_inicial = time()
                    cap = cv2.VideoCapture(0)

                    leido, frame = cap.read()
                    if leido == True:
                        cv2.imwrite("traffic.png", frame)
                        print("Captured photo")
                    else:
                        print("Error accessing camera")

                    cap.release()

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    img = cv2.imread("traffic.png")
                    cimg = img
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    # color range
                    lower_red1 = np.array([0,100,100])
                    upper_red1 = np.array([10,255,255])
                    lower_red2 = np.array([160,100,100])
                    upper_red2 = np.array([180,255,255])
                    lower_green = np.array([40,50,50])
                    upper_green = np.array([90,255,255])
                    # lower_yellow = np.array([15,100,100])
                    # upper_yellow = np.array([35,255,255])
                    lower_yellow = np.array([15,150,150])
                    upper_yellow = np.array([35,255,255])
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    maskg = cv2.inRange(hsv, lower_green, upper_green)
                    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
                    maskr = cv2.add(mask1, mask2)

                    size = img.shape
                    # print size

                    # hough circle detect
                    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                            param1=50, param2=10, minRadius=0, maxRadius=30)

                    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                                param1=50, param2=10, minRadius=0, maxRadius=30)

                    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                                param1=50, param2=5, minRadius=0, maxRadius=30)

                    # traffic light detect
                    r = 5
                    bound = 4.0 / 10
                    if r_circles is not None:
                        r_circles = np.uint16(np.around(r_circles))

                        for i in r_circles[0, :]:
                            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                                continue

                            h, s = 0.0, 0.0
                            for m in range(-r, r):
                                for n in range(-r, r):

                                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                                        continue
                                    h += maskr[i[1]+m, i[0]+n]
                                    s += 1
                            if h / s > 50:
                                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                                playsound('/home/sergio/Escritorio/Mobility_System/Audio/red_traffic_light.mp3')

                    if g_circles is not None:
                        g_circles = np.uint16(np.around(g_circles))

                        for i in g_circles[0, :]:
                            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                                continue

                            h, s = 0.0, 0.0
                            for m in range(-r, r):
                                for n in range(-r, r):

                                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                                        continue
                                    h += maskg[i[1]+m, i[0]+n]
                                    s += 1
                            if h / s > 100:
                                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                                playsound('/home/sergio/Escritorio/Mobility_System/Audio/green_traffic_light.mp3')

                    if y_circles is not None:
                        y_circles = np.uint16(np.around(y_circles))

                        for i in y_circles[0, :]:
                            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                                continue

                            h, s = 0.0, 0.0
                            for m in range(-r, r):
                                for n in range(-r, r):

                                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                                        continue
                                    h += masky[i[1]+m, i[0]+n]
                                    s += 1
                            if h / s > 50:
                                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                                playsound('/home/sergio/Escritorio/Mobility_System/Audio/yellow_traffic_light.mp3')

                    #cv2.imshow("Output", img)
                    #cv2.imwrite("semaforo2.jpg", img)
                    tiempo_final = time()
                    tiempo_ejecucion = tiempo_final - tiempo_inicial
                    print("Execution time ",tiempo_ejecucion)

                    #cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cont+=1

            if "location" in text:
                playsound('/home/sergio/Escritorio/Mobility_System/Audio/said_location.mp3')

                cont=1
                while (cont<=1):

                    running = True

                    def getPositionData(gps):
                        nx = gpsd.next()
                            
                        if nx['class'] == 'TPV':
                            latitude = getattr(nx,'lat', "Unknown")
                            longitude = getattr(nx,'lon', "Unknown")
                            print ("Tu posicion: lon = " + str(longitude) + ", lat = " + str(latitude))

                    gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)

                    try:
                        print ("Aplicacion iniciada!")
                        while running:
                            getPositionData(gpsd)
                            time.sleep(1.0)

                    except (KeyboardInterrupt):
                        running = False
                        print ("Aplicacion cerrada!")


                    def formatDegreesMinutes(coordinates, digits):
                        parts = coordinates.split(".")

                        if (len(parts) != 2):
                            return coordinates

                        if (digits > 3 or digits < 2):
                            return coordinates
                        
                        left = parts[0]
                        right = parts[1]
                        degrees = str(left[:digits])
                        minutes = str(right[:5])

                        return degrees + "." + minutes

                    geolocalizador=Nominatim(user_agent= 'Pruebas')
                    ubicacion=geolocalizador.reverse("20.391670, -99.994186")
                    ubi=ubicacion.address
                    print(ubi)


                    engine = pyttsx3.init('espeak')
                    engine.setProperty("rate",120)
                    engine.say('Your position ' +ubi)
                    #text = ("Tu posicion " +ubi)
                    #output_file = "audio.mp3"
                    #engine.save_to_file("text", "output_file")
                    engine.runAndWait()
                    cont+=1

            if "out" in text:
                playsound('/home/sergio/Escritorio/Mobility_System/Audio/system_off.mp3')
                print("The System is OFF")
                break
            
        except:
            playsound('/home/sergio/Escritorio/Mobility_System/Audio/repeat_please.mp3')
            print('Repeat, please')