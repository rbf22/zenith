from collections import Counter

from zenith.loop import sample_loops


def test_single_device_loop_distribution_is_uniform():
    counts = Counter(sample_loops(step, 0) for step in range(20000))
    assert set(counts) == {1, 2, 3, 4}
    assert all(abs(c / 20000 - 0.25) < 0.02 for c in counts.values())


def test_loop_count_is_shared_across_ranks_except_override():
    world = 10
    ks = [sample_loops(7, 0, rank=r, world_size=world) for r in range(world)]
    k = sample_loops(7, 0)
    expected = Counter({1: 1, k: world - 1}) if k != 1 else Counter({1: world})
    assert Counter(ks) == expected


def test_sampling_is_deterministic_for_same_step_and_micro():
    assert [sample_loops(s, m) for s in range(50) for m in range(3)] == [sample_loops(s, m) for s in range(50) for m in range(3)]


def test_no_override_when_fraction_is_zero():
    world = 8
    for step in range(20):
        assert {sample_loops(step, 0, rank=r, world_size=world, override_frac=0.0) for r in range(world)} == {sample_loops(step, 0)}


def test_override_rotates_across_steps():
    world = 10
    for step in range(100):
        base = sample_loops(step, 0)
        if base == 1:
            continue
        changed = {r for r in range(world) if sample_loops(step, 0, rank=r, world_size=world) != base}
        assert changed == {step % world}


def test_respects_max_loops():
    assert {sample_loops(s, 0, max_loops=2) for s in range(200)} == {1, 2}
