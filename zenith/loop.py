import random


def sample_loops(step, micro, rank=0, world_size=1, max_loops=4, override_frac=0.1):
    k = random.Random(f"{step}:{micro}").randint(1, max_loops)
    n_override = round(override_frac * world_size)
    offset = (step + micro) % world_size
    if (rank - offset) % world_size < n_override:
        return 1
    return k
