import cv2
from matplotlib import pyplot as plt

img = cv2.imread('humans/hum5.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

human_data = cv2.CascadeClassifier('cass/haarcascade_fullbody.xml')
humans = human_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
print(humans)

for(x, y, width, height) in humans:
    cv2.circle(img_rgb, (x + (width // 2), y + (height // 2)), width // 1, (0,255,0), 5)

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()

def check_foward(human_cords):
    if len(human_cords) != 0:
        return False
    else:
        return True

try:
    human_cords = human_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()
except:
    print('Людей на дороге нет')
    human_cords = []

print('Движение разрешено:', check_foward(human_cords))
