"""Four-channel XYZI, per-point binary MSG baseline; no structure losses."""

from models.pointnet2_sem_seg_msg import get_model as SemanticMSG


class get_model(SemanticMSG):
    def __init__(self):
        super().__init__(num_classes=2, input_channels=4)

    def forward(self, points):
        if points.ndim != 3 or points.shape[1] != 4 or points.shape[2] < 1024:
            raise ValueError("Expected [batch, 4, points] with at least 1024 points")
        return super().forward(points)
