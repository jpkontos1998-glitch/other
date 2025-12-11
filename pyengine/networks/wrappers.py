import torch
from torch.distributions import Categorical


class ClampModel(torch.nn.Module):
    def __init__(self, model, min_prob):
        super().__init__()
        self.model = model
        self.min_prob = min_prob

    def __call__(
        self,
        observations: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> torch.Tensor:
        tensor_dict = self.model(
            observations,
            piece_ids,
            legal_actions,
        )
        log_probs = tensor_dict["action_log_probs"]
        probs = log_probs.exp()
        probs[legal_actions] = probs[legal_actions].clamp_min(self.min_prob)
        probs[legal_actions] = probs[legal_actions] / probs[legal_actions].sum(dim=-1, keepdim=True)
        tensor_dict["action"] = Categorical(probs=probs).sample().int()
        return tensor_dict


class TemperatureModel(torch.nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def __call__(
        self,
        observations: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> torch.Tensor:
        tensor_dict = self.model(
            observations,
            piece_ids,
            legal_actions,
        )
        log_probs = tensor_dict["action_log_probs"] / self.temperature
        tensor_dict["actions"] = Categorical(logits=log_probs).sample().int()
        return tensor_dict


class ThresholdModel(torch.nn.Module):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def __call__(
        self,
        observations: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> torch.Tensor:
        tensor_dict = self.model(
            observations,
            piece_ids,
            legal_actions,
        )
        probs = tensor_dict["action_log_probs"].exp()
        mask = probs > self.threshold
        probs[~mask] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        action = Categorical(probs=probs).sample().int()
        tensor_dict["action"] = action
        return tensor_dict


class ArgmaxModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(
        self,
        observations: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> torch.Tensor:
        tensor_dict = self.model(
            observations,
            piece_ids,
            legal_actions,
        )
        tensor_dict["action"] = tensor_dict["action_log_probs"].argmax(dim=-1).int()
        return tensor_dict
