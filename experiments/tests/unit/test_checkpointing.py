import torch
from torch import nn

from tensorion_experiments import train


def test_save_checkpoint_roundtrip(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    inputs = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(inputs), labels)
    loss.backward()
    optimizer.step()
    scheduler.step()

    train.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=tmp_path,
        epoch=1,
    )

    epoch_dir = tmp_path / "epoch-1"
    model_file = epoch_dir / "model.pt"
    optim_file = epoch_dir / "optimizer.pt"
    sched_file = epoch_dir / "scheduler.pt"

    assert model_file.exists()
    assert optim_file.exists()
    assert sched_file.exists()

    reloaded_model = nn.Sequential(nn.Linear(4, 3))
    reloaded_optimizer = torch.optim.SGD(
        reloaded_model.parameters(), lr=0.1, momentum=0.9
    )
    reloaded_scheduler = torch.optim.lr_scheduler.StepLR(
        reloaded_optimizer, step_size=1, gamma=0.1
    )

    reloaded_model.load_state_dict(torch.load(model_file, map_location="cpu"))
    reloaded_optimizer.load_state_dict(torch.load(optim_file, map_location="cpu"))
    reloaded_scheduler.load_state_dict(torch.load(sched_file))

    for original, reloaded in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.allclose(original, reloaded)

    assert (
        reloaded_scheduler.state_dict()["last_epoch"]
        == scheduler.state_dict()["last_epoch"]
    )
