"""
DiffCal Interference Pipeline (Wave Superposition Fusion)

Based on diffcal_raw.py with bidirectional wave fusion:
- fusion_type='superposition' (WaveSuperposition)
- y(t) = alpha(t)*h_fwd + beta(t)*h_bwd + gamma(t)*(h_fwd * h_bwd)
- Position-dependent wave strength with interference term

Usage:
    python diffcal_interference.py runs --seeds='[1]' --masked_basin_ids 01022500
"""

from dataclasses import dataclass
from types import SimpleNamespace
import yaml
import os

from imputation.diffusion.src.experiments.diffcal_raw_res import DiffCalRaw, DiffCalRawParameters, dict2namespace
from src.models.NsDiff import NsDiff
import src.layer.mu_backbone as ns_Transformer


@dataclass
class DiffCalInterferenceParameters(DiffCalRawParameters):
    """Parameters for Interference (WaveSuperposition) fusion"""
    bidirectional: bool = True
    fusion_type: str = 'superposition'


@dataclass
class DiffCalInterference(DiffCalRaw, DiffCalInterferenceParameters):
    """
    DiffCal with Wave Superposition Fusion (Interference)

    Physical formula: y(t) = alpha(t)*h_fwd + beta(t)*h_bwd + gamma(t)*(h_fwd * h_bwd)
    - alpha(t): Forward wave strength (decreases with position)
    - beta(t): Backward wave strength (increases with position)
    - gamma(t): Interference term (strongest at middle)
    """
    model_type: str = "diffusion_interference"

    def _init_model(self):
        """Initialize model with bidirectional wave fusion"""
        self.label_len = self.windows // 2
        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features": 'MS',
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "enc_in": self.dataset.num_features,
            "dec_in": self.dataset.num_features,
            "c_out": 1,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_layers": self.d_layers,
            "d_ff": self.d_ff,
            "moving_avg": self.moving_avg,
            "timesteps": self.diffusion_steps,
            "factor": self.factor,
            "distil": self.distil,
            "beta_schedule": "linear",
            "embed": 'timeF',
            "dropout": self.dropout,
            "activation": self.activation,
            "output_attention": False,
            "do_predict": True,
            "k_z": self.k_z,
            "k_cond": self.k_cond,
            "p_hidden_dims": [64, 64],
            "freq": self.dataset.freq,
            "CART_input_x_embed_dim": self.CART_input_x_embed_dim,
            "p_hidden_layers": self.p_hidden_layers,
            "d_z": self.d_z,
            "diffusion_config_dir": "./configs/nsdiff.yml",
            "time_embed": False,
            # Wave fusion parameters
            "bidirectional": self.bidirectional,
            "fusion_type": self.fusion_type,
        }

        with open("./configs/nsdiff.yml", "r") as f:
            config = yaml.unsafe_load(f)
            self.diffusion_config = dict2namespace(config)

        self.args = SimpleNamespace(**args_dict)
        self.model = NsDiff(self.args, self.device).to(self.device)
        self.cond_pred_model = ns_Transformer.Model(self.args).float().to(self.device)
        self.cond_pred_model_g = None

        if self.load_pretrain:
            model_f_path = f"./results/runs/F/{self.dataset_type}/w{self.windows}h1s{self.pred_len}/1/best_model.pth"
            print("using pretrained model...")
            print(f"f(x): {model_f_path}")
            if os.path.exists(model_f_path):
                import torch
                self.cond_pred_model.load_state_dict(torch.load(model_f_path, map_location=self.device, weights_only=True))


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()

    fire.Fire(DiffCalInterference)
