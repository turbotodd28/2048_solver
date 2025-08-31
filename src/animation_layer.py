import random
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from game import Game2048

# ===== Animation event structures =====

@dataclass(frozen=True)
class MoveEvent:
    id: int
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    merges_into: Optional[int] = None  # target tile id if this move merges, else None

@dataclass(frozen=True)
class MergeEvent:
    into_row: int
    into_col: int
    from_ids: Tuple[int, int]  # (older/leftmost id, moving id) — stable ordering for deterministic visuals
    new_id: int
    new_value: int

@dataclass(frozen=True)
class SpawnEvent:
    row: int
    col: int
    value: int
    id: int


class AnimatedGame2048:
    """
    Animation layer that wraps the clean Game2048 class.
    Provides animation events while keeping the core game logic fast and simple.
    """
    
    def __init__(self, size: int = 4):
        self.game = Game2048(size)
        self._n = size
        # Animation-specific state
        self.id_grid = np.zeros((self._n, self._n), dtype=int)  # stable tile IDs for animation
        self._next_id = 1
        
        # Initialize IDs for existing tiles
        self._assign_ids_to_existing_tiles()
    
    def _fresh_id(self) -> int:
        """Generate a fresh tile ID."""
        nid = self._next_id
        self._next_id += 1
        return nid
    
    def _assign_ids_to_existing_tiles(self):
        """Assign IDs to tiles that already exist on the board."""
        for r in range(self._n):
            for c in range(self._n):
                if self.game.board[r, c] != 0 and self.id_grid[r, c] == 0:
                    self.id_grid[r, c] = self._fresh_id()
    
    def _tile_pos(self, tid: int) -> Optional[Tuple[int, int]]:
        """Find the current (row, col) of a tile id; None if not present."""
        if tid == 0:
            return None
        locs = np.argwhere(self.id_grid == tid)
        if locs.size == 0:
            return None
        r, c = locs[0]
        return int(r), int(c)
    
    def _map_coords_inverse(self, r: int, c: int, direction: str, rotated: bool) -> Tuple[int, int]:
        """Map coordinates from transformed space back to original grid."""
        if direction in ['down', 'right']:
            c = self._n - 1 - c
        if rotated:
            r, c = c, r
        return int(r), int(c)
    
    def _map_coords_forward(self, r: int, c: int, direction: str) -> Tuple[int, int]:
        """Map original coordinates (r,c) to transformed space."""
        rr, cc = r, c
        rotated = False
        if direction in ['up', 'down']:
            rr, cc = cc, rr
            rotated = True
        if direction in ['down', 'right']:
            cc = self._n - 1 - cc
        return int(rr), int(cc)
    
    def plan_move(self, direction: str) -> Tuple[List[MoveEvent], List[MergeEvent], Optional[SpawnEvent]]:
        """
        Compute animation events for a move without modifying the game state.
        Returns (move_events, merge_events, spawn_event or None).
        """
        if direction not in ['up', 'down', 'left', 'right']:
            raise ValueError("Invalid move direction")

        # Take transformed copies for left-merge computation
        vboard, rotated = self._transform(self.game.board.copy(), direction)
        vid, _ = self._transform(self.id_grid.copy(), direction)

        move_events: List[MoveEvent] = []
        merge_events: List[MergeEvent] = []

        # We will build new rows into these arrays
        new_vals = np.zeros_like(vboard)
        new_ids = np.zeros_like(vid)

        for r in range(self._n):
            vals = vboard[r].copy()
            ids = vid[r].copy()

            dest = 0
            merged_flag = [False] * self._n

            # For each source col c in left-to-right order
            for c in range(self._n):
                if vals[c] == 0:
                    continue
                v = int(vals[c])
                tid = int(ids[c])

                if dest > 0 and new_vals[r, dest - 1] == v and not merged_flag[dest - 1]:
                    # Merge into dest-1
                    target_id = int(new_ids[r, dest - 1])
                    from_pos = self._tile_pos(tid)
                    dest_r_orig, dest_c_orig = self._map_coords_inverse(r, dest - 1, direction, rotated)

                    # Record the move of the moving tile
                    if from_pos is None:
                        from_row = from_col = -1
                    else:
                        from_row, from_col = from_pos

                    move_events.append(MoveEvent(
                        id=tid,
                        from_row=from_row, from_col=from_col,
                        to_row=dest_r_orig, to_col=dest_c_orig,
                        merges_into=target_id
                    ))

                    # Create the new merged tile id
                    new_tid = self._fresh_id()
                    new_val = v * 2

                    # For deterministic visuals, order the "from_ids"
                    pos_target = self._tile_pos(target_id)
                    pos_mover = from_pos
                    if pos_target is None or pos_mover is None:
                        ordered = (target_id, tid)
                    else:
                        ordered = tuple(sorted([target_id, tid],
                                               key=lambda t: self._tile_pos(t)))

                    # Record merge event
                    into_r_orig, into_c_orig = dest_r_orig, dest_c_orig
                    merge_events.append(MergeEvent(
                        into_row=into_r_orig,
                        into_col=into_c_orig,
                        from_ids=(ordered[0], ordered[1]),
                        new_id=new_tid,
                        new_value=new_val
                    ))

                    # Place merged result into the building row
                    new_vals[r, dest - 1] = new_val
                    new_ids[r, dest - 1] = new_tid
                    merged_flag[dest - 1] = True

                else:
                    # Simple slide to 'dest'
                    from_pos = self._tile_pos(tid)
                    dest_r_orig, dest_c_orig = self._map_coords_inverse(r, dest, direction, rotated)

                    # Record move only if position actually changes
                    if from_pos is not None:
                        from_row, from_col = from_pos
                        if (from_row != dest_r_orig) or (from_col != dest_c_orig):
                            move_events.append(MoveEvent(
                                id=tid,
                                from_row=from_row, from_col=from_col,
                                to_row=dest_r_orig, to_col=dest_c_orig,
                                merges_into=None
                            ))

                    new_vals[r, dest] = v
                    new_ids[r, dest] = tid
                    dest += 1

        # If no moves and no merges, nothing changes and there should be no spawn
        if not move_events and not merge_events:
            return [], [], None

        # Build the future logical state to pick a spawn
        fvals = self._inverse_transform(new_vals, direction, rotated)
        fids = self._inverse_transform(new_ids, direction, rotated)

        # Choose a spawn in that future state
        empties = [(i, j) for i in range(self._n) for j in range(self._n) if fvals[i, j] == 0]
        spawn_event: Optional[SpawnEvent] = None
        if empties:
            si, sj = random.choice(empties)
            sval = 2 if random.random() < 0.9 else 4
            sid = self._fresh_id()
            spawn_event = SpawnEvent(si, sj, sval, sid)

        return move_events, merge_events, spawn_event

    def apply_planned_result(self,
                           move_events: List[MoveEvent],
                           merge_events: List[MergeEvent],
                           spawn: Optional[SpawnEvent],
                           direction: str) -> None:
        """
        Apply the planned result to both the game state and animation state.
        """
        # Apply the move logic directly to avoid double tile spawning
        self._apply_move_to_game(move_events, merge_events, spawn, direction)
        
        # Update the animation ID grid
        self._update_id_grid_after_move(move_events, merge_events, spawn, direction)
    
    def _apply_move_to_game(self,
                           move_events: List[MoveEvent],
                           merge_events: List[MergeEvent],
                           spawn: Optional[SpawnEvent],
                           direction: str) -> None:
        """
        Apply the planned move result directly to the game state without spawning a random tile.
        """
        # Transform board for left-merge computation
        vboard, rotated = self._transform(self.game.board.copy(), direction)
        
        # Track score gain
        total_score_gain = 0

        # Process each row
        for r in range(self._n):
            old_row = vboard[r].copy()
            new_row, score_gain = self._slide_and_merge_row(old_row)
            vboard[r] = new_row
            total_score_gain += score_gain

        # Transform back to original orientation
        self.game.board = self._inverse_transform(vboard, direction, rotated)
        self.game.score += total_score_gain
        self.game.move_count += 1

        # Apply the planned spawn (not a random one)
        if spawn is not None and spawn.row >= 0:
            self.game.board[spawn.row, spawn.col] = spawn.value

    def _slide_and_merge_row(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Slide and merge a single row, return (new_row, score_gain)."""
        non_zero = row[row != 0]
        out = []
        score_gain = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = non_zero[i] * 2
                out.append(merged_val)
                score_gain += merged_val
                i += 2
            else:
                out.append(non_zero[i])
                i += 1
        # Pad with zeros
        while len(out) < len(row):
            out.append(0)
        return np.array(out, dtype=int), score_gain

    def _update_id_grid_after_move(self,
                                 move_events: List[MoveEvent],
                                 merge_events: List[MergeEvent],
                                 spawn: Optional[SpawnEvent],
                                 direction: str) -> None:
        """Update the ID grid to match the new game state by applying planned events directly."""
        # Track which tiles are involved in this move
        involved_tiles = set()
        
        # Collect all tile IDs involved in moves
        for me in move_events:
            involved_tiles.add(me.id)
        
        # Collect all tile IDs involved in merges
        for me in merge_events:
            involved_tiles.update(me.from_ids)
            involved_tiles.add(me.new_id)
        
        # Collect spawn tile ID
        if spawn is not None and spawn.row >= 0:
            involved_tiles.add(spawn.id)
        
        # Only clear positions that are involved in this move
        # This preserves stationary tiles that didn't move
        for me in move_events:
            current_pos = self._tile_pos(me.id)
            if current_pos is not None:
                self.id_grid[current_pos[0], current_pos[1]] = 0
        
        for me in merge_events:
            for source_id in me.from_ids:
                current_pos = self._tile_pos(source_id)
                if current_pos is not None:
                    self.id_grid[current_pos[0], current_pos[1]] = 0
        
        # Apply move events to update positions
        for me in move_events:
            self.id_grid[me.to_row, me.to_col] = me.id
        
        # Apply merge events
        for me in merge_events:
            self.id_grid[me.into_row, me.into_col] = me.new_id
        
        # Apply spawn event
        if spawn is not None and spawn.row >= 0:
            self.id_grid[spawn.row, spawn.col] = spawn.id

    def move(self, direction: str) -> Tuple[List[MoveEvent], List[MergeEvent], Optional[SpawnEvent]]:
        """
        Perform a move and return animation events.
        Raises ValueError on invalid/no-op move.
        """
        move_events, merge_events, spawn = self.plan_move(direction)
        if not move_events and not merge_events:
            raise ValueError("Invalid move: No tiles moved or combined.")
        
        # Apply the planned result
        self.apply_planned_result(move_events, merge_events, spawn, direction)
        return move_events, merge_events, spawn

    def _transform(self, arr: np.ndarray, direction: str) -> Tuple[np.ndarray, bool]:
        """Return a transformed view for computing 'left' logic."""
        rotated = False
        out = arr
        if direction in ['up', 'down']:
            out = out.T
            rotated = True
        if direction in ['down', 'right']:
            out = np.flip(out, axis=1)
        return out, rotated

    def _inverse_transform(self, arr: np.ndarray, direction: str, rotated: bool) -> np.ndarray:
        out = arr
        if direction in ['down', 'right']:
            out = np.flip(out, axis=1)
        if rotated:
            out = out.T
        return out

    # ===== Delegate methods to the core game =====
    
    def get_state(self) -> np.ndarray:
        return self.game.get_state()
    
    def get_score(self) -> int:
        return self.game.get_score()
    
    def get_move_count(self) -> int:
        return self.game.get_move_count()
    
    def get_valid_moves(self) -> List[str]:
        return self.game.get_valid_moves()
    
    def is_game_over(self) -> bool:
        return self.game.is_game_over()
    
    def reset(self):
        self.game.reset()
        self.id_grid = np.zeros((self._n, self._n), dtype=int)
        self._next_id = 1
        self._assign_ids_to_existing_tiles()
    
    @property
    def board(self):
        return self.game.board
    
    @property
    def n(self):
        return self._n
