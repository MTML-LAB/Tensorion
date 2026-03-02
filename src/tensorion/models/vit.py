import torch
import torch.nn as nn


def VIT_0(num_classes: int = 101):
    """Legacy FasterViT model (kept for backward compatibility)."""
    from fastervit import create_model

    model = create_model("faster_vit_1_224")
    model.head = nn.Linear(640, num_classes)
    return model


def create_timm_model(
    model_name: str,
    num_classes: int = 1000,
    pretrained: bool = True,
    img_size: int | None = None,
    **kwargs,
) -> nn.Module:
    """Create a timm model by name.

    This is a unified function that can create any timm model by providing
    the model name. The model name should match the timm model registry.

    Args:
        model_name: Name of the timm model (e.g., 'vit_base_patch16_rope_mixed_ape_224.naver_in1k')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        img_size: Input image size. If None, uses model default. If provided, overrides model default.
        **kwargs: Additional arguments to pass to timm.create_model

    Returns:
        PyTorch model instance

    Examples:
        >>> model = create_timm_model('vit_base_patch16_rope_mixed_ape_224.naver_in1k', num_classes=10)
        >>> model = create_timm_model('swin_tiny_patch4_window7_224.ms_in1k', num_classes=100, pretrained=False, img_size=224)
    """
    import timm

    # If img_size is provided, pass it to timm
    if img_size is not None:
        kwargs["img_size"] = img_size

    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes, **kwargs
    )
    return model


def get_default_img_size(model_name: str) -> int | None:
    """Get the default image size for a given timm model name.

    Args:
        model_name: Name of the timm model

    Returns:
        Default image size if known, None otherwise
    """
    # Extract img_size from model name (e.g., "224" from "vit_base_patch16_rope_mixed_ape_224.naver_in1k")
    import re

    match = re.search(r"_(\d+)\.", model_name)
    if match:
        return int(match.group(1))

    # Fallback: check known model sizes
    known_sizes = {
        "swin_tiny_patch4_window7_224.ms_in1k": 224,
        "swin_small_patch4_window7_224.ms_in22k": 224,
        "swin_base_patch4_window7_224.ms_in22k": 224,
        "swin_large_patch4_window12_384.ms_in22k": 384,
        "vit_little_patch16_reg1_gap_256.sbb_in12k_ft_in1k": 256,
        "vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k": 256,
        "vit_base_patch16_rope_mixed_ape_224.naver_in1k": 224,
        "vit_large_patch16_rope_mixed_ape_224.naver_in1k": 224,
        "faster_vit_1_224": 224,
    }

    return known_sizes.get(model_name, None)


def get_hidden_dim(model_name: str) -> int:
    """Get the hidden dimension for a given timm model name.

    Args:
        model_name: Name of the timm model

    Returns:
        Hidden dimension (embedding size) of the model, or None if unknown
    """
    timm_model_dims = {
        "swin_tiny_patch4_window7_224.ms_in1k": 768,
        "swin_small_patch4_window7_224.ms_in22k": 768,
        "swin_base_patch4_window7_224.ms_in22k": 1024,
        "swin_large_patch4_window12_384.ms_in22k": 1536,
        "vit_little_patch16_reg1_gap_256.sbb_in12k_ft_in1k": 320,
        "vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k": 512,
        "vit_base_patch16_rope_mixed_ape_224.naver_in1k": 768,
        "vit_large_patch16_rope_mixed_ape_224.naver_in1k": 1024,
        "faster_vit_1_224": 640,  # VIT_0
    }

    return timm_model_dims.get(model_name, None)


def infer_hidden_dim(model: nn.Module) -> int:
    """Infer the hidden dimension from a model by inspecting its architecture.

    Args:
        model: PyTorch model

    Returns:
        Hidden dimension if found, None otherwise
    """
    # Try to get from model attributes (common in timm models)
    if hasattr(model, "num_features"):
        return model.num_features
    if hasattr(model, "embed_dim"):
        return model.embed_dim
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        return model.head.in_features

    # Try to get from classifier/head layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            "head" in name.lower() or "classifier" in name.lower()
        ):
            return module.in_features

    # Try forward_features if available (timm models)
    if hasattr(model, "forward_features"):
        try:
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)
                features = model.forward_features(x)
                if isinstance(features, torch.Tensor):
                    return features.shape[-1]
                elif isinstance(features, (list, tuple)):
                    return features[-1].shape[-1]
        except:
            pass

    return None
