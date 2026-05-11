import os
import sys
import random
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pygame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from rl.q_agent import QAgent

# =========================================================
# CONFIG
# =========================================================
WIDTH, HEIGHT = 800, 600
FPS = 60

PADDLE_W, PADDLE_H = 15, 140
BALL_SIZE = 15
PADDLE_SPEED = 7

BALL_SPEED_X = 4
BALL_SPEED_Y = 4

# ---------------- FEEDBACK & DIFFICULTY ----------------
feedback_text = ""
feedback_timer = 0

base_paddle_speed = PADDLE_SPEED
base_ball_speed_x = BALL_SPEED_X
base_ball_speed_y = BALL_SPEED_Y
WIN_THRESHOLD = 3

# =========================================================
# PATHS
# =========================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_DIR, "data", "models", "gesture_model.pkl")

# =========================================================
# INIT GAME
# =========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Adaptive AI Pong (CV)")
clock = pygame.time.Clock()

# Fonts — use a system font that supports ASCII well
font       = pygame.font.SysFont("Arial", 28)
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_small = pygame.font.SysFont("Arial", 20)

# =========================================================
# COLORS
# =========================================================
WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
YELLOW     = (255, 220,  50)
CYAN       = ( 80, 220, 255)
RED        = (255,  80,  80)
GRAY       = (160, 160, 160)
DARK_GRAY  = ( 30,  30,  30)
OVERLAY    = (  0,   0,   0, 160)  # semi-transparent

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# =========================================================
# MEDIAPIPE
# =========================================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(0)

# =========================================================
# OBJECTS
# =========================================================
player = pygame.Rect(30, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
ai     = pygame.Rect(WIDTH - 45, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
ball   = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)

ball_speed_x = random.choice([-BALL_SPEED_X, BALL_SPEED_X])
ball_speed_y = random.choice([-BALL_SPEED_Y, BALL_SPEED_Y])

player_score = 0
ai_score     = 0

# =========================================================
# PAUSE STATE
# =========================================================
paused = False

# =========================================================
# Q-LEARNING AGENT
# =========================================================
actions = ["UP", "DOWN", "STAY"]
agent   = QAgent(actions=actions)
agent.load()

# =========================================================
# DIFFICULTY LEVEL LABEL
# =========================================================
difficulty_label = "Normal"

# =========================================================
# HELPERS
# =========================================================
def reset_ball():
    global ball_speed_x, ball_speed_y
    ball.center = (WIDTH // 2, HEIGHT // 2)
    # Use the CURRENT speed magnitudes so difficulty is respected after reset
    spd_x = abs(ball_speed_x) if ball_speed_x != 0 else BALL_SPEED_X
    spd_y = abs(ball_speed_y) if ball_speed_y != 0 else BALL_SPEED_Y
    ball_speed_x = random.choice([-spd_x, spd_x])
    ball_speed_y = random.choice([-spd_y, spd_y])


def get_state():
    relative_y  = (ball.centery - ai.centery) // 15
    distance_x  = abs(ball.centerx - ai.centerx) // 30
    direction_y = 1 if ball_speed_y > 0 else -1
    return [relative_y, distance_x, direction_y]


def apply_difficulty(score_diff):
    """Return (paddle_speed, ball_sx, ball_sy, label) based on score gap."""
    global PADDLE_SPEED, BALL_SPEED_X, BALL_SPEED_Y

    if score_diff >= WIN_THRESHOLD:
        ps   = base_paddle_speed + 2
        bsx  = base_ball_speed_x + 1
        bsy  = base_ball_speed_y + 1
        lbl  = "HARD"
    elif score_diff <= -WIN_THRESHOLD:
        ps   = base_paddle_speed
        bsx  = base_ball_speed_x
        bsy  = base_ball_speed_y
        lbl  = "Normal"
    else:
        ps   = base_paddle_speed
        bsx  = base_ball_speed_x
        bsy  = base_ball_speed_y
        lbl  = "Normal"

    PADDLE_SPEED = ps
    BALL_SPEED_X = bsx
    BALL_SPEED_Y = bsy
    return lbl


def scale_ball_speed(new_sx, new_sy):
    """Keep ball direction but update magnitude to new_sx / new_sy."""
    global ball_speed_x, ball_speed_y
    ball_speed_x = new_sx * (1 if ball_speed_x >= 0 else -1)
    ball_speed_y = new_sy * (1 if ball_speed_y >= 0 else -1)


def draw_dashed_center_line():
    """Vertical dashed line in the middle of the court."""
    dash_h = 18
    gap    = 10
    x      = WIDTH // 2 - 1
    y      = 0
    while y < HEIGHT:
        pygame.draw.rect(screen, DARK_GRAY, (x, y, 2, dash_h))
        y += dash_h + gap


def draw_paddle(rect, color=WHITE):
    """Rounded-rectangle paddle."""
    pygame.draw.rect(screen, color, rect, border_radius=6)


def draw_ball(rect, color=WHITE):
    pygame.draw.ellipse(screen, color, rect)


def draw_hud():
    # Score
    score_surf = font_large.render(f"{player_score}   {ai_score}", True, WHITE)
    screen.blit(score_surf, (WIDTH // 2 - score_surf.get_width() // 2, 14))

    # Labels under score
    you_surf = font_small.render("YOU", True, GRAY)
    ai_surf  = font_small.render("AI",  True, GRAY)
    screen.blit(you_surf, (WIDTH // 2 - score_surf.get_width() // 2 - 4, 64))
    screen.blit(ai_surf,  (WIDTH // 2 + score_surf.get_width() // 2 - ai_surf.get_width() + 4, 64))

    # Difficulty badge
    diff_color = RED if difficulty_label == "HARD" else CYAN
    diff_surf  = font_small.render(f"[ {difficulty_label} ]", True, diff_color)
    screen.blit(diff_surf, (WIDTH // 2 - diff_surf.get_width() // 2, HEIGHT - 28))

    # Pause hint
    hint = font_small.render("P = Pause", True, DARK_GRAY)
    screen.blit(hint, (WIDTH - hint.get_width() - 10, HEIGHT - 24))


def draw_feedback():
    if feedback_text:
        # Subtle pill background
        surf = font.render(feedback_text, True, YELLOW)
        pad  = 12
        bg   = pygame.Surface((surf.get_width() + pad * 2, surf.get_height() + pad), pygame.SRCALPHA)
        bg.fill((50, 40, 0, 180))
        bx = WIDTH // 2 - bg.get_width() // 2
        by = 90
        screen.blit(bg,   (bx, by))
        screen.blit(surf, (bx + pad, by + pad // 2))


def draw_pause_overlay():
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 155))
    screen.blit(overlay, (0, 0))

    title = font_large.render("PAUSED", True, WHITE)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 60))

    sub = font.render("Press P to resume  |  Q to quit", True, GRAY)
    screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 10))


# =========================================================
# MAIN LOOP
# =========================================================
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            agent.save()
            cap.release()
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused = not paused
            if event.key == pygame.K_q:
                agent.save()
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()

    # ---- If paused: draw overlay and skip update ----
    if paused:
        screen.fill(BLACK)
        draw_dashed_center_line()
        draw_paddle(player)
        draw_paddle(ai, CYAN)
        draw_ball(ball)
        draw_hud()
        draw_pause_overlay()
        pygame.display.flip()
        clock.tick(FPS)
        continue

    # ---------------- CAMERA / MODEL ----------------
    ret, frame = cap.read()
    label = "NONE"

    if ret:
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                features = []
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - wrist_x)
                    features.append(lm.y - wrist_y)
                    features.append(lm.z)

                X     = np.array(features).reshape(1, -1)
                label = model.predict(X)[0]

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if label == "UP":
            player.y -= PADDLE_SPEED
        elif label == "DOWN":
            player.y += PADDLE_SPEED

        player.y = max(0, min(HEIGHT - PADDLE_H, player.y))

        cv2.putText(frame, f"Gesture: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "P = Pause  Q = Quit", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Webcam Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            agent.save()
            break

    # ---------------- AI ----------------
    state         = get_state()
    learning_phase = (ball_speed_x > 0)

    if learning_phase:
        action = agent.choose_action(state)
    else:
        action = "STAY"

    if action == "UP":
        ai.y -= PADDLE_SPEED
    elif action == "DOWN":
        ai.y += PADDLE_SPEED

    ai.y = max(0, min(HEIGHT - PADDLE_H, ai.y))

    # ---------------- BALL ----------------
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    reward = 0

    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1

    if ball.colliderect(player):
        ball.left     = player.right
        ball_speed_x *= -1

    if ball.colliderect(ai):
        ball.right    = ai.left
        ball_speed_x *= -1
        reward       += 10

    distance  = abs(ai.centery - ball.centery)
    reward   -= distance * 0.01

    if ball.left <= 0:
        ai_score += 1
        reward   += 5
        reset_ball()

    if ball.right >= WIDTH:
        player_score += 1
        reward       -= 20
        reset_ball()

    next_state = get_state()
    if learning_phase:
        agent.update(state, action, reward, next_state)

    if agent.epsilon > 0.02:
        agent.epsilon *= 0.9995

    # ---------------- FEEDBACK ----------------
    # Replace emojis with ASCII-safe text symbols
    if reward >= 10:
        feedback_text  = ">> Nice move! <<"
        feedback_timer = 60
    elif reward <= -10:
        feedback_text  = "!! You are predictable !!"
        feedback_timer = 60

    if feedback_timer > 0:
        feedback_timer -= 1
    else:
        feedback_text = ""

    # ---------------- ADAPT DIFFICULTY ----------------
    score_diff       = player_score - ai_score
    prev_bsx         = abs(ball_speed_x)
    prev_bsy         = abs(ball_speed_y)
    difficulty_label = apply_difficulty(score_diff)

    # FIX: propagate new speed magnitudes to the live ball
    if abs(ball_speed_x) != BALL_SPEED_X or abs(ball_speed_y) != BALL_SPEED_Y:
        scale_ball_speed(BALL_SPEED_X, BALL_SPEED_Y)

    # ---------------- DRAW ----------------
    screen.fill(BLACK)
    draw_dashed_center_line()
    draw_paddle(player)
    draw_paddle(ai, CYAN)
    draw_ball(ball)
    draw_hud()
    draw_feedback()

    pygame.display.flip()
    clock.tick(FPS)

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()