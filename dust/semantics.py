"""Official raw IDs; binary keep/dust/ignore mapping remains in dust.data.

Source: https://github.com/vincentweikey/LiDARDustX#semantic-segmentation-label-definition
Checked 2026-10-20. Names describe annotations, not automatically verified truth.
"""

SEMANTIC_SOURCE = "https://github.com/vincentweikey/LiDARDustX#semantic-segmentation-label-definition"
SEMANTIC_NAMES = dict(enumerate((
    "Ignore", "Ground", "Wall", "ArtificialWall", "Mountain", "Tree", "Grass",
    "VerticalObstacle", "StaticObstacle", "Dust", "Noise", "Rainwater", "Car",
    "Truck", "Engineering Vehicle", "Bus", "Pickup", "Pedestrian", "Rider",
    "Tricycle", "Bicycle", "Construction", "Trailer", "Wire", "Animal", "Unknown",
    "Stone", "Mound", "Panel",
)))
