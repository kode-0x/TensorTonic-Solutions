import numpy as np

def resolve_go_action(board: list, player: int, action: int, consecutive_passes: int, move_count: int) -> tuple:
    """
    Returns: a tuple: next board, player, captures, consecutive passes, move count, terminal.
    """
    b = np.array(board, dtype=int, copy=True)
    n = b.shape[0]
    captures = 0

    if action == n * n:
        consecutive_passes += 1
    else:
        consecutive_passes = 0
        r, c = divmod(action, n)
        b[r, c] = player

        def group_and_liberties(sr, sc):
            stack = [(sr, sc)]
            group = set()
            liberties = set()

            while stack:
                x, y = stack.pop()

                if (x, y) in group:
                    continue

                group.add((x, y))

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy

                    if not (0 <= nx < n and 0 <= ny < n):
                        continue

                    if b[nx, ny] == 0:
                        liberties.add((nx, ny))
                    elif b[nx, ny] == b[sr, sc] and (nx, ny) not in group:
                        stack.append((nx, ny))

            return group, liberties

        checked = set()

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dx, c + dy

            if not (0 <= nr < n and 0 <= nc < n):
                continue

            if b[nr, nc] != -player or (nr, nc) in checked:
                continue

            group, liberties = group_and_liberties(nr, nc)
            checked.update(group)

            if not liberties:
                captures += len(group)

                for x, y in group:
                    b[x, y] = 0

    move_count += 1
    next_player = -player
    terminal = consecutive_passes >= 2 or move_count >= 2 * n * n

    return b, next_player, captures, consecutive_passes, move_count, terminal