import cv2

print("Opening camera...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        print("FAILED TO READ FRAME")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
