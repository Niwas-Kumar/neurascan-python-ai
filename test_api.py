import requests
import numpy as np
import cv2

# Create a dummy image with some text
img = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, "The qvick brwon f0x jumpps ov3r the lazzy d0g", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.imwrite("test_sample.png", img)

# Send to the API
url = 'http://localhost:5000/analyze'
files = {'file': open('test_sample.png', 'rb')}
response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
