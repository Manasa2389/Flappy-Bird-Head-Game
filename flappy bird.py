import cv2
import mediapipe as mp
import numpy as np
import random

class Bird:
    def __init__(self, x, y, radius):
        self.x, self.y = x, y
        self.radius = radius
        self.lerp_factor = 0.4 

    def update(self, target_y):
        self.y += (target_y - self.y) * self.lerp_factor
        self.y = max(self.radius, min(self.y, 720 - 70 - self.radius))

    def draw(self, img):
        # Shadow
        cv2.ellipse(img, (int(self.x), 670), (25, 10), 0, 0, 360, (60, 100, 60), -1)
        # Body
        cv2.circle(img, (int(self.x), int(self.y)), self.radius, (0, 200, 255), -1)
        cv2.circle(img, (int(self.x), int(self.y)), self.radius, (0, 0, 0), 2)
        # Eye
        cv2.circle(img, (int(self.x + 10), int(self.y - 10)), 6, (255, 255, 255), -1)
        cv2.circle(img, (int(self.x + 12), int(self.y - 10)), 3, (0, 0, 0), -1)
        # Beak
        pts = np.array([[self.x+20, self.y-5], [self.x+40, self.y+5], [self.x+20, self.y+15]], np.int32)
        cv2.fillPoly(img, [pts], (0, 50, 255))

class AdvancedHeadGame:
    def __init__(self):
        self.width, self.height = 1280, 720
        self.BIRD_X, self.PIPE_WIDTH, self.PIPE_GAP = 200, 100, 240
        self.BASE_SPEED = 15
        self.SENSITIVITY = 2.2 
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.reset_game()

    def reset_game(self):
        self.bird = Bird(self.BIRD_X, self.height // 2, 28)
        self.pipes, self.score, self.frame_count = [], 0, 0
        self.game_active = self.game_over = False
        self.target_y = self.height // 2

    def get_head_pos(self, frame):
        small_rgb = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(small_rgb)
        if results.multi_face_landmarks:
            nose_bridge = results.multi_face_landmarks[0].landmark[6].y
            offset = nose_bridge - 0.5
            mapped_y = 0.5 + (offset * self.SENSITIVITY)
            self.target_y = int(mapped_y * self.height)
            return results.multi_face_landmarks[0]
        return None

    def run(self):
        win_name = "Advanced Head Control"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        while True:
            success, frame = self.cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            face_landmarks = self.get_head_pos(frame)
            
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            canvas[:] = (245, 230, 220) # BG Color

            if self.game_active and not self.game_over:
                self.frame_count += 1
                self.bird.update(self.target_y)
                
                if self.frame_count % 30 == 0:
                    self.pipes.append([self.width, random.randint(100, 400)])
                
                for p in self.pipes:
                    p[0] -= self.BASE_SPEED
                    # Collision
                    if p[0] < self.bird.x + 20 < p[0] + self.PIPE_WIDTH:
                        if self.bird.y - 20 < p[1] or self.bird.y + 20 > p[1] + self.PIPE_GAP:
                            self.game_over = True
                    # Score
                    if p[0] + self.PIPE_WIDTH < 0:
                        self.pipes.pop(0)
                        self.score += 1

            # --- Drawing Pipes ---
            for p in self.pipes:
                cv2.rectangle(canvas, (int(p[0]), 0), (int(p[0]+self.PIPE_WIDTH), p[1]), (70, 180, 70), -1)
                cv2.rectangle(canvas, (int(p[0]), p[1]+self.PIPE_GAP), (int(p[0]+self.PIPE_WIDTH), self.height-70), (70, 180, 70), -1)

            # Ground and Bird
            cv2.rectangle(canvas, (0, self.height-70), (self.width, self.height), (100, 150, 100), -1)
            self.bird.draw(canvas)

            # --- Face Preview ---
            if face_landmarks:
                preview_roi = cv2.resize(frame[100:380, 200:440], (200, 150))
                self.mp_drawing.draw_landmarks(preview_roi, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None, connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
                canvas[20:170, self.width-220:self.width-20] = preview_roi

            # --- UI & Game Over Screen ---
            cv2.putText(canvas, f"SCORE: {self.score}", (40, 80), 1, 3, (50, 50, 50), 3)

            if not self.game_active and not self.game_over:
                cv2.putText(canvas, "SPACE TO START", (self.width//2-200, self.height//2), 2, 1.5, (100,100,100), 2)

            if self.game_over:
                # 1. Create a darkened overlay
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 150), -1)
                alpha = 0.4  # Transparency factor
                canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
                
                # 2. Game Over Text
                cv2.putText(canvas, "GAME OVER", (self.width//2-220, self.height//2-50), 2, 2.5, (255, 255, 255), 5)
                cv2.putText(canvas, f"FINAL SCORE: {self.score}", (self.width//2-160, self.height//2 + 30), 2, 1.2, (255, 255, 255), 2)
                cv2.putText(canvas, "PRESS 'R' TO RESTART", (self.width//2-190, self.height//2 + 100), 2, 1, (200, 200, 200), 2)

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(1)
            if key == ord('q'): break
            elif key == 32: self.game_active = True
            elif key == ord('r'): self.reset_game()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    AdvancedHeadGame().run() 