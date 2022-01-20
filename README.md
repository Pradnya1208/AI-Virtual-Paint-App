<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">AI Virtual Paint App</div>
<div align="center"><img src="https://github.com/Pradnya1208/AI-Virtual-Paint-App/blob/main/output/intro.gif?raw=true" width="40%"></div>



## Overview:
In this project, we are going to create a virtual painter using AI. We will first track our hand and get its landmarks and then use the points to draw on the screen. We will use two fingers for selection and one finger for drawing. And the best part is that all of this will be done in real-time.


## Implementation:

**Libraries:**  `NumPy` `pandas` `sklearn` `mediapipe` `cv2` `Matplotlib`

## Hands tracking:
The Mediapipe hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates inside the detected hand regions via regression, that is direct coordinate prediction. The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions.
<br>
<img src="https://github.com/Pradnya1208/AI-Virtual-Paint-App/blob/main/output/hands%20tracking.PNG?raw=true">
#### Code snippets:
```
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
```
```
def findHands(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    
    if self.results.multi_hand_landmarks:
        for handLms in self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, handLms,
                self.mpHands.HAND_CONNECTIONS)
 
    return img
```
```
cap = cv2.VideoCapture(1)
detector = handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
```
We get all the hand landmarks with above code.
<img src="https://github.com/Pradnya1208/AI-Virtual-Paint-App/blob/main/output/landamrks.PNG?raw=true">
<br>
#### Getting the position values of hand landmraks:
```
def findPosition(self, img, handNo=0, draw=True):
    xList = []
    yList = []
    bbox = []
    self.lmList = []
    if self.results.multi_hand_landmarks:
        myHand = self.results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            # print(id, cx, cy)
            self.lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
    xmin, xmax = min(xList), max(xList)
    ymin, ymax = min(yList), max(yList)
    bbox = xmin, ymin, xmax, ymax
 
    if draw:
        cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
        (0, 255, 0), 2)
 
    return self.lmList, bbox
```
`lmlist` in above function wll return all the landamrk positions.

## Virtual Painter:
We have designed the painting options using Canva. Checkout the design [here](https://www.canva.com/design/DAEyy-fWQ_g/share/preview?token=SGajZM-7HExP7_7pL4h6ow&role=EDITOR&utm_content=DAEyy-fWQ_g&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton).
Then, we overlayed these option son our capture window, to give it an effect of Paint app.
<br>
Following are the steps for handtracking and drawing.
- Import the image
- Find the hand landmarks using Hand Tracking Module
```
img = detector.findHands(img)
lmList = detector.findPosition(img, draw=False)
```
- Check which fingers are up (for the seoaration between selection, drawing and erasing)
```
fingers = detector.fingersUp()
```
- If selection mode - 2 fingers are up
```
if fingers[1] and fingers[2]:
// check for the clicks
```
- If drawing mode -  Index fingure is up 
```
if fingers[1] and fingers[2] == False:
cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
```

Check the [implementation](https://github.com/Pradnya1208/AI-Virtual-Paint-App/blob/main/VirtualPainter.py) for more details.

## Result:

<div align="center"><img src="https://github.com/Pradnya1208/AI-Virtual-Paint-App/blob/main/outputgif.gif?raw=true" width="60%"></div>
<br>

### Learnings:
`Computer vision`
`Landamark detection`
`Hands movement tracking using Mediapipe and cv2`







## References:
[mediapipe](https://mediapipe.dev/)
[hands tracking](https://google.github.io/mediapipe/solutions/hands.html)
### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner



[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

