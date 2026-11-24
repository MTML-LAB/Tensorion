import torch

from tensorion.models.unet import FCNHead, UNetBackbone, UNetEncoderDecoder


def test_unet_fcn_forward_and_slide_inference_shapes():
    model = UNetEncoderDecoder(
        backbone=UNetBackbone(
            in_channels=3,
            base_channels=16,
            num_stages=5,
            strides=(1, 1, 1, 1, 1),
            enc_num_convs=(1, 1, 1, 1, 1),
            dec_num_convs=(1, 1, 1, 1),
            downsamples=(True, True, True, True),
            enc_dilations=(1, 1, 1, 1, 1),
            dec_dilations=(1, 1, 1, 1),
            norm_cfg={"type": "BN"},
            act_cfg={"type": "ReLU"},
        ),
        decode_head=FCNHead(
            in_channels=16,
            in_index=4,
            channels=16,
            num_convs=1,
            concat_input=False,
            num_classes=2,
            align_corners=False,
            loss_weight=1.0,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "ReLU"},
        ),
        auxiliary_head=FCNHead(
            in_channels=32,
            in_index=3,
            channels=16,
            num_convs=1,
            concat_input=False,
            num_classes=2,
            align_corners=False,
            loss_weight=0.4,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "ReLU"},
        ),
        test_cfg={"mode": "slide", "crop_size": (32, 32), "stride": (21, 21)},
    )

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out["decode"].shape == (2, 2, 64, 64)
    assert out["aux"].shape == (2, 2, 64, 64)

    slide = model.inference(x)
    assert slide.shape == (2, 2, 64, 64)
