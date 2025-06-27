#export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib && python3 engines.py
import chess
import chess.engine
import time
import pygame
import os
import cairosvg

# Path to Stockfish binary
STOCKFISH_PATH = "/Users/jatin/Documents/python/supervisedchess/stockfish/stockfish-macos-m1-apple-silicon"  # update this

# Pygame setup
pygame.init()
SQUARE_SIZE = 60
WIDTH, HEIGHT = SQUARE_SIZE * 8, SQUARE_SIZE * 8
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stockfish vs Stockfish")

colors = [pygame.Color("blue"), pygame.Color("gray")]

PIECE_IMAGES = {}
PIECE_NAMES = {
    'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
}

def load_piece_images():
    base_path = "/Users/jatin/Documents/python/supervisedchess/pieces"  # Change if needed
    for symbol, name in PIECE_NAMES.items():
        svg_file = os.path.join(base_path, f"{name}.svg")
        png_file = os.path.join(base_path, f"{name}.png")

        # Convert SVG to PNG if needed
        if not os.path.exists(png_file):
            cairosvg.svg2png(url=svg_file, write_to=png_file, output_width=SQUARE_SIZE, output_height=SQUARE_SIZE)

        image = pygame.image.load(png_file)
        PIECE_IMAGES[symbol] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(board):
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(file*SQUARE_SIZE, rank*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            square = chess.square(file, 7 - rank)
            piece = board.piece_at(square)
            if piece:
                img = PIECE_IMAGES[piece.symbol()]
                screen.blit(img, (file*SQUARE_SIZE, rank*SQUARE_SIZE))
    pygame.display.flip()

def square_to_pixel(square):
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square)
    return file * SQUARE_SIZE, rank * SQUARE_SIZE

# Animate a piece moving from one square to another
def animate_move(board, move, piece_symbol):
    start = square_to_pixel(move.from_square)
    end = square_to_pixel(move.to_square)
    frames = 15
    dx = (end[0] - start[0]) / frames
    dy = (end[1] - start[1]) / frames
    for i in range(1, frames + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        draw_board(board)
        x = start[0] + dx * i
        y = start[1] + dy * i
        img = PIECE_IMAGES[piece_symbol]
        screen.blit(img, (x, y))
        pygame.display.flip()
        clock.tick(60)
    return True

# Initialize board
board = chess.Board()

# Start Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

load_piece_images()

print("Stockfish vs. Stockfish begins...\n")

running = True
clock = pygame.time.Clock()

while running and not board.is_game_over():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_board(board)
    time.sleep(0.5)  # Shorter pause before move

    result = engine.play(board, chess.engine.Limit(time=0.1))
    move = result.move
    piece = board.piece_at(move.from_square)
    # Animate the move before updating the board
    if not animate_move(board, move, piece.symbol()):
        running = False
        break
    board.push(move)
    print(f"Engine plays: {move}\n")
    time.sleep(0.5)  # Shorter pause after move

print("\nGame Over")
print("Result:", board.result())

engine.quit()
pygame.quit()
