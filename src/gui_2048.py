import math
import sys
import pygame
from typing import Dict, Tuple, Optional, List
from animation_layer import AnimatedGame2048, MoveEvent, MergeEvent, SpawnEvent

# ------------------------
# Config
# ------------------------
GRID_N = 4
TILE_SIZE = 110
GAP = 12
BORDER = 18
HUD_HEIGHT = 120
WINDOW_W = BORDER * 2 + GRID_N * TILE_SIZE + (GRID_N - 1) * GAP
WINDOW_H = HUD_HEIGHT + BORDER * 2 + GRID_N * TILE_SIZE + (GRID_N - 1) * GAP
FPS = 60

# Animation timings (ms)
SLIDE_MS_PER_CELL = 50         # slide scales with Manhattan distance in cells
SLIDE_MIN_MS = 50
MERGE_POP_MS = 90
SPAWN_MS = 90

# Colors
BG_COLOR = (250, 248, 239)
BOARD_BG = (187, 173, 160)
EMPTY_TILE = (205, 193, 180)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)

# Classic-ish palette
VALUE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
# Fallback for > 2048
def color_for(v: int) -> Tuple[int, int, int]:
    if v in VALUE_COLORS:
        return VALUE_COLORS[v]
    # simple fade for larger numbers
    t = min(1.0, math.log2(max(2048, v)) - 11)  # 2048 -> 0, 4096 -> 1
    base = (60, 58, 50)
    return (int(237*(1-t) + base[0]*t), int(194*(1-t) + base[1]*t), int(46*(1-t) + base[2]*t))

# ------------------------
# Helpers
# ------------------------
def grid_to_px(r: int, c: int) -> Tuple[int, int]:
    x = BORDER + c * (TILE_SIZE + GAP)
    y = HUD_HEIGHT + BORDER + r * (TILE_SIZE + GAP)
    return x, y

def ease_out_cubic(t: float) -> float:
    return 1 - pow(1 - t, 3)

def ease_out_back(t: float, s: float = 1.70158) -> float:
    # pleasing pop
    t -= 1
    return (t * t * ((s + 1) * t + s) + 1)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ------------------------
# Sprite for a single logical tile id
# ------------------------
class TileSprite:
    def __init__(self, tid: int, value: int, row: int, col: int):
        self.id = tid
        self.value = value
        self.row = row
        self.col = col
        self.x, self.y = grid_to_px(row, col)
        self.scale = 1.0
        self.alpha = 255
        self.visible = True

        # animation targets
        self._anim_active = False
        self._start_pos = (self.x, self.y)
        self._end_pos = (self.x, self.y)
        self._start_time = 0
        self._duration = 1
        self._on_done = None

    def set_grid_pos(self, row: int, col: int):
        self.row, self.col = row, col
        self.x, self.y = grid_to_px(row, col)

    def start_slide(self, end_row: int, end_col: int, duration_ms: int, now_ms: int, on_done=None):
        self._anim_active = True
        self._start_pos = (self.x, self.y)
        self._end_pos = grid_to_px(end_row, end_col)
        self._start_time = now_ms
        self._duration = max(1, duration_ms)
        self._on_done = on_done

    def update(self, now_ms: int):
        if not self._anim_active:
            return
        t = clamp01((now_ms - self._start_time) / self._duration)
        e = ease_out_cubic(t)
        sx, sy = self._start_pos
        ex, ey = self._end_pos
        self.x = sx + (ex - sx) * e
        self.y = sy + (ey - sy) * e
        if t >= 1.0:
            self._anim_active = False
            # Snap to end
            self.x, self.y = ex, ey
            if self._on_done:
                cb = self._on_done
                self._on_done = None
                cb()

# ------------------------
# Animation controller
# ------------------------
class MoveAnimation:
    """Owns the move -> pop merges -> spawn timeline."""
    def __init__(self,
                 game: AnimatedGame2048,
                 sprites: Dict[int, TileSprite],
                 move_events: List[MoveEvent],
                 merge_events: List[MergeEvent],
                 spawn: Optional[SpawnEvent],
                 direction: str,
                 now_ms: int):
        self.game = game
        self.sprites = sprites
        self.move_events = move_events
        self.merge_events = merge_events
        self.spawn_ev = spawn
        self.direction = direction

        self.phase = "slide"
        self.phase_start = now_ms
        self.phase_done = False

        # Keep track of slide completion
        self.sliding_ids = set()
        longest_ms = 0
        for m in move_events:
            spr = sprites.get(m.id)
            if not spr:
                continue
            dist_cells = abs(m.to_row - m.from_row) + abs(m.to_col - m.from_col)
            dur = max(SLIDE_MIN_MS, dist_cells * SLIDE_MS_PER_CELL)
            longest_ms = max(longest_ms, dur)
            self.sliding_ids.add(m.id)
            # If this tile will merge, we still slide it to the merge target
            spr.start_slide(m.to_row, m.to_col, dur, now_ms, on_done=lambda tid=m.id: self._on_slide_done(tid))

        # Tiles that are not moving stay put; merged-into targets may also be stationary.
        self.expected_slide_done = set(self.sliding_ids)

        # For the merge pop phase we create "future" merged sprites (hidden initially)
        self.future_merged_sprites: Dict[int, TileSprite] = {}
        for me in merge_events:
            tid = me.new_id
            r, c = me.into_row, me.into_col
            merged = TileSprite(tid, me.new_value, r, c)
            merged.scale = 0.9
            merged.alpha = 0  # invisible until pop
            self.future_merged_sprites[tid] = merged

        # Spawn sprite placeholder
        self.future_spawn_sprite: Optional[TileSprite] = None
        if spawn is not None and spawn.row >= 0:
            s = TileSprite(spawn.id, spawn.value, spawn.row, spawn.col)
            s.scale = 0.6
            s.alpha = 0
            self.future_spawn_sprite = s

    def _on_slide_done(self, tid: int):
        self.sliding_ids.discard(tid)

    def update(self, now_ms: int):
        if self.phase == "slide":
            # Update all sprites doing slides
            for spr in list(self.sprites.values()):
                spr.update(now_ms)

            # When all moving tiles finished sliding, hide merged sources and advance
            if not self.sliding_ids:
                # Hide the two sources that merged; keep them in dict for now, but invisible
                merged_sources = set()
                for me in self.merge_events:
                    a, b = me.from_ids
                    merged_sources.add(a)
                    merged_sources.add(b)
                for tid in merged_sources:
                    spr = self.sprites.get(tid)
                    if spr:
                        spr.visible = False
                        # Track hidden sprites for cleanup
                        if not hasattr(self, 'hidden_sprites'):
                            self.hidden_sprites = set()
                        self.hidden_sprites.add(tid)

                # Advance to merge pop
                self.phase = "merge_pop"
                self.phase_start = now_ms
                # Insert future merged sprites into the render dict so they can pop
                for tid, spr in self.future_merged_sprites.items():
                    self.sprites[tid] = spr

        elif self.phase == "merge_pop":
            t = clamp01((now_ms - self.phase_start) / MERGE_POP_MS)
            e = ease_out_back(t)
            for me in self.merge_events:
                spr = self.sprites.get(me.new_id)
                if spr:
                    spr.scale = 0.9 + (1.08 - 0.9) * e
                    spr.alpha = int(255 * t)
            if t >= 1.0:
                # settle scale at 1.0
                for me in self.merge_events:
                    spr = self.sprites.get(me.new_id)
                    if spr:
                        spr.scale = 1.0
                        spr.alpha = 255
                # Next phase
                self.phase = "spawn"
                self.phase_start = now_ms
                # Remove merged source sprites for real
                for me in self.merge_events:
                    a, b = me.from_ids
                    if a in self.sprites:
                        self.sprites.pop(a, None)
                    if b in self.sprites:
                        self.sprites.pop(b, None)

        elif self.phase == "spawn":
            if self.future_spawn_sprite is None:
                # Commit logic immediately if no spawn
                self._commit_and_sync()
                self.phase = "done"
                return
            t = clamp01((now_ms - self.phase_start) / SPAWN_MS)
            e = ease_out_cubic(t)
            spr = self.future_spawn_sprite
            spr.scale = 0.6 + (1.0 - 0.6) * e
            spr.alpha = int(255 * t)
            # ensure it's in render dict
            self.sprites[spr.id] = spr
            if t >= 1.0:
                spr.scale = 1.0
                spr.alpha = 255
                # Commit logic then finish
                self._commit_and_sync()
                self.phase = "done"

    def _commit_and_sync(self):
        # Apply the logical result to the game
        self.game.apply_planned_result(self.move_events, self.merge_events, self.spawn_ev, self.direction)
        
        # Build set of ids currently on board
        ids_on_board = set(int(i) for i in self.game.id_grid.flatten() if int(i) != 0)
        
        # Remove sprites that are not on the board or are invisible
        for tid in list(self.sprites.keys()):
            spr = self.sprites[tid]
            if tid not in ids_on_board or not spr.visible:
                self.sprites.pop(tid, None)
        
        # Clean up any tracked hidden sprites
        if hasattr(self, 'hidden_sprites'):
            for tid in self.hidden_sprites:
                if tid in self.sprites:
                    self.sprites.pop(tid, None)
            self.hidden_sprites.clear()
        
        # Ensure there is a sprite for every id in the grid and it's snapped to cell
        for r in range(GRID_N):
            for c in range(GRID_N):
                tid = int(self.game.id_grid[r, c])
                val = int(self.game.board[r, c])
                if tid == 0:
                    continue
                spr = self.sprites.get(tid)
                if spr is None:
                    spr = TileSprite(tid, val, r, c)
                    self.sprites[tid] = spr
                else:
                    spr.value = val
                    spr.set_grid_pos(r, c)
                    spr.visible = True
                    spr.scale = 1.0
                    spr.alpha = 255

    def is_done(self) -> bool:
        return self.phase == "done"

# ------------------------
# Rendering
# ------------------------
def draw_board(surface: pygame.Surface):
    surface.fill(BG_COLOR)
    # HUD box
    pygame.draw.rect(surface, BOARD_BG, pygame.Rect(BORDER, BORDER, WINDOW_W - 2 * BORDER, HUD_HEIGHT - BORDER), border_radius=10)

    # Grid background
    board_rect = pygame.Rect(BORDER, HUD_HEIGHT, WINDOW_W - 2 * BORDER, WINDOW_H - HUD_HEIGHT - BORDER)
    pygame.draw.rect(surface, BOARD_BG, board_rect, border_radius=12)

    # Empty cells
    for r in range(GRID_N):
        for c in range(GRID_N):
            x, y = grid_to_px(r, c)
            pygame.draw.rect(surface, EMPTY_TILE, pygame.Rect(x, y, TILE_SIZE, TILE_SIZE), border_radius=8)

def draw_hud(surface: pygame.Surface, game: AnimatedGame2048, font_title: pygame.font.Font, font_small: pygame.font.Font):
    # Title
    title = font_title.render("2048", True, TEXT_DARK)
    surface.blit(title, (BORDER + 4, BORDER - 2))

    # Score boxes
    def small_box(label: str, value: str, x: int):
        w = 140
        h = 56
        rect = pygame.Rect(x, BORDER + 8, w, h)
        pygame.draw.rect(surface, (187,173,160), rect, border_radius=8)
        lab = font_small.render(label, True, TEXT_LIGHT)
        val = font_small.render(value, True, TEXT_LIGHT)
        surface.blit(lab, (rect.x + 12, rect.y + 8))
        surface.blit(val, (rect.x + 12, rect.y + 28))

    small_box("SCORE", str(game.get_score()), WINDOW_W - BORDER - 140)
    small_box("MOVES", str(game.get_move_count()), WINDOW_W - BORDER - 140 - 12 - 140)

def draw_tiles(surface: pygame.Surface, sprites: Dict[int, TileSprite], font_big: pygame.font.Font, font_med: pygame.font.Font):
    # Draw in order: lower alpha behind higher for nice overlap
    for tid, spr in sorted(sprites.items(), key=lambda kv: (kv[1].alpha, kv[0])):
        if not spr.visible:
            continue
        rect = pygame.Rect(0, 0, int(TILE_SIZE * spr.scale), int(TILE_SIZE * spr.scale))
        rect.center = (spr.x + TILE_SIZE // 2, spr.y + TILE_SIZE // 2)
        color = color_for(spr.value)
        tile_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(tile_surface, color + (spr.alpha,), pygame.Rect(0, 0, rect.width, rect.height), border_radius=8)

        # choose font size
        text = str(spr.value)
        font = font_big if spr.value < 1024 else font_med
        text_color = TEXT_DARK if spr.value <= 4 else TEXT_LIGHT
        text_surf = font.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=(rect.width // 2, rect.height // 2))
        tile_surface.blit(text_surf, text_rect)

        surface.blit(tile_surface, rect.topleft)



# ------------------------
# Main loop
# ------------------------
def build_initial_sprites(game: AnimatedGame2048) -> Dict[int, TileSprite]:
    sprites: Dict[int, TileSprite] = {}
    for r in range(GRID_N):
        for c in range(GRID_N):
            tid = int(game.id_grid[r, c])
            val = int(game.board[r, c])
            if tid != 0:
                sprites[tid] = TileSprite(tid, val, r, c)
    return sprites

def main():
    pygame.init()
    pygame.display.set_caption("2048 (Animated)")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    # fonts
    font_title = pygame.font.SysFont("arial", 56, bold=True)
    font_big = pygame.font.SysFont("arial", 40, bold=True)
    font_med = pygame.font.SysFont("arial", 32, bold=True)
    font_small = pygame.font.SysFont("arial", 20, bold=True)

    game = AnimatedGame2048(size=GRID_N)
    sprites = build_initial_sprites(game)
    current_anim: Optional[MoveAnimation] = None

    running = True
    while running:
        now_ms = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Accept input only if no animation in progress
            if event.type == pygame.KEYDOWN and (current_anim is None):
                key = event.key
                dir_map = {
                    pygame.K_UP: "up", pygame.K_w: "up",
                    pygame.K_DOWN: "down", pygame.K_s: "down",
                    pygame.K_LEFT: "left", pygame.K_a: "left",
                    pygame.K_RIGHT: "right", pygame.K_d: "right",
                }
                if key in dir_map:
                    direction = dir_map[key]
                    try:
                        move_events, merge_events, spawn = game.plan_move(direction)
                        if move_events or merge_events:
                            current_anim = MoveAnimation(game, sprites, move_events, merge_events, spawn, direction, now_ms)
                    except Exception as e:
                        print(f"Move error: {e}")

                # quick reset
                if key == pygame.K_r:
                    game = AnimatedGame2048(size=GRID_N)
                    sprites = build_initial_sprites(game)
                    current_anim = None

        # Update animation
        if current_anim is not None:
            current_anim.update(now_ms)
            if current_anim.is_done():
                current_anim = None

        # Draw
        draw_board(screen)
        draw_hud(screen, game, font_title, font_small)
        draw_tiles(screen, sprites, font_big, font_med)

        # Game over overlay
        if game.is_game_over() and current_anim is None:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 180))
            screen.blit(overlay, (0, 0))
            msg = font_title.render("Game Over", True, TEXT_DARK)
            screen.blit(msg, msg.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
