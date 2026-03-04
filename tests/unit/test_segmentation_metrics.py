import torch

from tensorion.segmentation_metrics import DiceMeter


def test_dice_meter_perfect_prediction_gives_one():
    meter = DiceMeter(num_classes=2, ignore_index=255)

    logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [10.0, -10.0]],
                [[-10.0, 10.0], [-10.0, 10.0]],
            ]
        ]
    )
    target = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)

    meter.update(logits, target)
    result = meter.compute()

    assert result["mDice"] == 1.0
    assert result["dice_class_0"] == 1.0
    assert result["dice_class_1"] == 1.0
