from planar_physics.contract import VARIANT

TRAINING = {
    '3r': dict(mixed=2048, mixed_seed=3_100_000, wall=1536, wall_seed=5_100_000),
    '2r': dict(mixed=3000, mixed_seed=105_100_000, wall=1750, wall_seed=105_500_000),
}[VARIANT]
SELECTION = {'3r': (69_200_000, 96), '2r': (106_200_000, 96)}[VARIANT]
FINAL = {'3r': (90_000_000, 512), '2r': (108_000_000, 256)}[VARIANT]
