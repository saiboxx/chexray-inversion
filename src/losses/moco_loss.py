import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn


class MocoLoss(nn.Module):
    def __init__(self, model_path: str = 'models/mocov2.pt') -> None:
        super().__init__()
        self.model_path = model_path
        print('Loading MOCO model from path: {}'.format(self.model_path))
        self.model = self._load_model(model_path)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _load_model(model_path: str) -> nn.Module:

        model = torchvision.models.__dict__['resnet50']()
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith(
                'module.encoder_q.fc'
            ):
                # remove prefix
                state_dict[k[len('module.encoder_q.') :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}
        # remove output layer
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def extract_feats(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, size=224)
        x_feats = self.model(x)
        x_feats = nn.functional.normalize(x_feats, dim=1)
        x_feats = x_feats.squeeze(-1).squeeze(-1)
        return x_feats

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        n_samples = x.shape[0]
        with torch.no_grad():
            x_feats = self.extract_feats(x)
        x_hat_feats = self.extract_feats(x_hat)

        loss = 0.0
        for i in range(n_samples):
            diff_target = x_hat_feats[i].dot(x_feats[i])
            loss += 1 - diff_target

        return loss / n_samples
