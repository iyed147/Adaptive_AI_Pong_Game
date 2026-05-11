import pygame
import sys
import random

from rl.q_agent import QAgent

# ---------------- CONFIG ----------------

WIDTH = 800
HEIGHT = 600

FPS = 60

PADDLE_W = 15
PADDLE_H = 140

BALL_SIZE = 15

PADDLE_SPEED = 7

BALL_SPEED_X = 4
BALL_SPEED_Y = 4

AUTO_TRAINING = False

# ---------------- INIT ----------------

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Adaptive AI Pong"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "Arial",
    28
)

# ---------------- OBJETS ----------------

player = pygame.Rect(
    30,
    HEIGHT // 2 - PADDLE_H // 2,
    PADDLE_W,
    PADDLE_H
)

ai = pygame.Rect(
    WIDTH - 45,
    HEIGHT // 2 - PADDLE_H // 2,
    PADDLE_W,
    PADDLE_H
)

ball = pygame.Rect(
    WIDTH // 2 - BALL_SIZE // 2,
    HEIGHT // 2 - BALL_SIZE // 2,
    BALL_SIZE,
    BALL_SIZE
)

ball_speed_x = random.choice(
    [-BALL_SPEED_X, BALL_SPEED_X]
)

ball_speed_y = random.choice(
    [-BALL_SPEED_Y, BALL_SPEED_Y]
)

# ---------------- SCORES ----------------

player_score = 0
ai_score = 0

# ---------------- IA ----------------

actions = [
    "UP",
    "DOWN",
    "STAY"
]

agent = QAgent(
    actions=actions
)

agent.load()

# ---------------- FONCTIONS ----------------

def reset_ball():

    global ball_speed_x
    global ball_speed_y

    ball.center = (
        WIDTH // 2,
        HEIGHT // 2
    )

    ball_speed_x = random.choice(
        [-BALL_SPEED_X, BALL_SPEED_X]
    )

    ball_speed_y = random.choice(
        [-BALL_SPEED_Y, BALL_SPEED_Y]
    )


def get_state():

    # État simplifié et intelligent
def get_state():

    # Distance verticale balle / IA
    relative_y = (
        ball.centery - ai.centery
    ) // 15

    # Distance horizontale
    distance_x = abs(
        ball.centerx - ai.centerx
    ) // 30

    # Direction verticale balle
    direction_y = (
        1 if ball_speed_y > 0 else -1
    )

    # Direction horizontale balle
    direction_x = (
        1 if ball_speed_x > 0 else -1
    )

    return [

        relative_y,

        distance_x,

        direction_y,

        direction_x
    ]

def move_player():

    if AUTO_TRAINING:

        # IA simple pour entraîner
        if player.centery < ball.centery:
            player.y += PADDLE_SPEED

        elif player.centery > ball.centery:
            player.y -= PADDLE_SPEED

    else:

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            player.y -= PADDLE_SPEED

        if keys[pygame.K_DOWN]:
            player.y += PADDLE_SPEED

    player.y = max(
        0,
        min(
            HEIGHT - PADDLE_H,
            player.y
        )
    )

# ---------------- BOUCLE PRINCIPALE ----------------

while True:

    # -------- EVENTS --------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            agent.save()

            pygame.quit()
            sys.exit()

    # -------- JOUEUR --------

    move_player()

    # -------- IA --------

    state = get_state()

    learning_phase = (
        ball_speed_x > 0
    )

    if learning_phase:

        action = agent.choose_action(
            state
        )

    else:

        action = "STAY"

    # Mouvement IA
    if action == "UP":

        ai.y -= PADDLE_SPEED

    elif action == "DOWN":

        ai.y += PADDLE_SPEED

    ai.y = max(
        0,
        min(
            HEIGHT - PADDLE_H,
            ai.y
        )
    )

    # -------- BALLE --------

    ball.x += ball_speed_x
    ball.y += ball_speed_y

    reward = 0

    # Collision murs
    if (
        ball.top <= 0
        or ball.bottom >= HEIGHT
    ):

        ball_speed_y *= -1

    # Collision joueur
    if ball.colliderect(player):

        ball.left = player.right

        ball_speed_x *= -1

    # Collision IA
    if ball.colliderect(ai):

        ball.right = ai.left

        ball_speed_x *= -1

        reward += 10

    # Punition distance
    distance = abs(
        ai.centery - ball.centery
    )

    reward -= distance * 0.01

    # Point IA
    if ball.left <= 0:

        ai_score += 1

        reward += 5

        reset_ball()

    # Point joueur
    if ball.right >= WIDTH:

        player_score += 1

        reward -= 20

        reset_ball()

    # -------- APPRENTISSAGE --------

    next_state = get_state()

    if learning_phase:

        agent.update(
            state,
            action,
            reward,
            next_state
        )

    # Décroissance epsilon
    if agent.epsilon > 0.02:

        agent.epsilon *= 0.9995

    # -------- AFFICHAGE --------

    screen.fill((0, 0, 0))

    pygame.draw.rect(
        screen,
        (255, 255, 255),
        player
    )

    pygame.draw.rect(
        screen,
        (255, 255, 255),
        ai
    )

    pygame.draw.ellipse(
        screen,
        (255, 255, 255),
        ball
    )

    # Texte score
    score_text = font.render(

        f"{player_score} - {ai_score}",

        True,

        (255, 255, 255)
    )

    screen.blit(

        score_text,

        (
            WIDTH // 2
            - score_text.get_width() // 2,

            20
        )
    )

    # Affichage epsilon
    epsilon_text = font.render(

        f"Epsilon: {agent.epsilon:.3f}",

        True,

        (180, 180, 180)
    )

    screen.blit(
        epsilon_text,
        (20, 20)
    )

    pygame.display.flip()

    clock.tick(FPS)