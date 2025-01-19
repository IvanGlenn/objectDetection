import face_recognition as FR
import cv2


class FaceRecognition:
    face_positions = []
    face_encodings = []
    face_tags = []
    process_current_frame = True

    def __init__(self):
        self.run_recognition()

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()

            # Resize the frames
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # OpenCV uses BGR and not RGB for some reason
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces in the current frame
            self.face_positions = FR.face_locations(rgb_small_frame)

            self.face_tags = []
            self.face_tags.extend([f'Face {k + 1}' for k in range(len(self.face_positions))])

            # Display annotations. The 'zip' function combines face_positions and face_tags lists and allows them to
            # be iterated over in parallel.
            for (t, r, b, l), name in zip(self.face_positions, self.face_tags):
                # Bring frame back to its original dimensions
                t *= 4
                r *= 4
                b *= 4
                l *= 4

                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), thickness=2)
                # Create another rectangle to hold the text. -1 fills the square instead of giving it a thickness
                cv2.rectangle(frame, (l, b - 35), (r, b), (0, 255, 0), thickness=-1)
                cv2.putText(frame, name, (l + 6, b - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), thickness=1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    # fr.run_recognition()
