import cv2
import pytesseract

# Load image
image = cv2.imread('product_image.jpg')

# Preprocess image (Grayscale, blur, threshold)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, threshold_img = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

# Extract text using Tesseract
text = pytesseract.image_to_string(threshold_img)

# Output the extracted text
print("Extracted Text:", text)
