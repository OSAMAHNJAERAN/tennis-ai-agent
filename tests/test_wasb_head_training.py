import copy
import pytest
import torch
from scripts.train.finetune_wasb_head import configure_head, frozen_state_equal, validate_reused_clip


def test_head_update_preserves_frozen_features_and_batchnorm_buffers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.BatchNorm1d(3))
            self.final_layers=torch.nn.Linear(3,1)
        def forward(self,x):
            return self.final_layers(self.backbone(x))
    model=Model()
    original=copy.deepcopy(model.state_dict())
    assert configure_head(model)==['final_layers.weight','final_layers.bias']
    assert not model.training and not model.backbone[1].training
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    loss=(model(torch.ones(4,2))-2).square().mean()
    loss.backward()
    optimizer.step()
    assert frozen_state_equal(model,original)
    assert not torch.equal(model.final_layers.bias,original['final_layers.bias'])
    model.backbone[1].running_mean[0]+=1
    assert not frozen_state_equal(model,original)


def test_cached_selection_requires_exact_target_and_windows():
    clip=dict(width=1920,height=1080,fps=60.,frames=20,stride=2,
              rows=[dict(frame=9,target_xy=[10.,11.],windows=[[[5,7,9],2],[[7,9,11],1],[[9,11,13],0]])])
    rows=[dict(Frame='9',Visibility='1',X='10',Y='11')]
    validate_reused_clip(clip,rows,1920,1080,60.,20)
    bad=copy.deepcopy(clip)
    bad['rows'][0]['target_xy'][0]+=1
    with pytest.raises(ValueError,match='target'):
        validate_reused_clip(bad,rows,1920,1080,60.,20)
    bad=copy.deepcopy(clip)
    bad['rows'][0]['windows'][0][0][0]+=1
    with pytest.raises(ValueError,match='context'):
        validate_reused_clip(bad,rows,1920,1080,60.,20)
    with pytest.raises(ValueError,match='geometry'):
        validate_reused_clip(clip,rows,1280,720,60.,20)


def test_no_head_model_rejected():
    with pytest.raises(ValueError,match='prediction head'):
        configure_head(torch.nn.Linear(2,1))
