# """
# Author : Sunil Kumar
# Description : OCR
# """
from PIL import Image,ImageOps
import numpy as np
import pytesseract
import cv2


def remove_noise_and_smooth(img):
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    or_image = cv2.bitwise_or(img, closing)
    return or_image

def textCheck(imgData):
# 	"""
# 	Input : Image 
# 	output : text found in image
# 	"""
	ret,img = cv2.threshold(np.array(imgData), 125, 255, cv2.THRESH_BINARY)

	#remove noise
	img=remove_noise_and_smooth(img)
	new_image = Image.fromarray(img.astype(np.uint8))

	# padding and expansion	
	new_image=ImageOps.expand(new_image,(0,0,0,10),(256,256,256))
	new_image = Image.fromarray(np.array(new_image)).convert('L') #now need gray scale

	basewidth = 900

	new_image = new_image.resize((900,47), Image.ANTIALIAS)

	text = pytesseract.image_to_string(new_image)

	return text.strip()

