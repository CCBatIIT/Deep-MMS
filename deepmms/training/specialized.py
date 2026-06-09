"""
Specialized Experiment subclasses for novel molecular autoencoder architectures.

Each trainer overrides the gradient step logic to accommodate the specific
loss landscape of its model family:

BetaVAETrainer  – adds beta-weighted KL penalty to the RMSD loss
VQVAETrainer    – adds commitment loss from vector quantization
HVAETrainer     – sums KL divergences from both hierarchical levels
FlowTrainer     – minimises NLL = 0.5*||z||^2 - log_det (no RMSD)
MAETrainer      – computes RMSD only on randomly masked atoms

All trainers write RMSD to the ``RMSD_Loss_Term`` NetCDF variable and use the
existing orbax checkpoint infrastructure from the parent Experiment class.
"""

import jax
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import orbax_utils

from .trainer import Experiment
from .loss import KL_loss, atom_rmsd, give_weighted_rmsd_func
from ..models.beta_vae import BetaVAE
from ..models.vq_vae import VQVAE
from ..models.hierarchical_vae import HierarchicalVAE
from ..models.flow_vae import RealNVPFlow
from ..models.mae_vae import MaskedAutoencoder


class BetaVAETrainer(Experiment):
    """
    Trainer for BetaVAE: RMSD loss + beta-weighted KL divergence.

    The step function minimises:
        log(RMS RMSD) + beta * KL_loss(z_mean, z_logvar)

    Parameters
    ----------
    json_fn : str or dict
        JSON config path or pre-loaded parameter dict.
    make_dirs : bool
        Create output directories if needed.
    from_json_params : bool
        When True treat json_fn as a dict.
    model_cls : type
        Model class (default BetaVAE).
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False, model_cls=BetaVAE):
        super().__init__(
            json_fn,
            make_dirs=make_dirs,
            from_json_params=from_json_params,
            model_cls=model_cls,
        )
        # Re-compile step with beta-augmented loss
        self._step, self._evaluate = self._build_beta_step()

    def _build_beta_step(self):
        """
        Build JIT-compiled step and evaluate closures with beta KL penalty.

        Returns
        -------
        step : callable
        evaluate : callable
        """
        atom_rmsd_loss = self._atom_rmsd_loss if hasattr(self, "_atom_rmsd_loss") \
            else give_weighted_rmsd_func(jnp.ones(self.model.input_size // 3))

        is_bn = self.is_batchnorm
        model_ref = self.model

        if is_bn:
            @jax.jit
            def step(state, batch_x, z_rng, dropout_key):
                """Beta-VAE gradient step with BatchNorm."""
                dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

                def loss_fn(params):
                    logits, updates = state.apply_fn(
                        {"params": params, "batch_stats": state.batch_stats},
                        batch_x, z_rng, train=True,
                        rngs={"dropout": dropout_train_key},
                        mutable=["batch_stats"],
                    )
                    decoded, z_mean, z_logvar = logits
                    rmsd = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)))
                    kl = KL_loss(z_mean, z_logvar)
                    beta = model_ref.beta
                    loss = rmsd + beta * kl
                    return loss, (logits, updates)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, (logits, updates)), grads = grad_fn(state.params)
                state = state.apply_gradients(grads=grads)
                state = state.replace(batch_stats=updates["batch_stats"])
                return state, loss

            @jax.jit
            def evaluate(state, batch_x, z_rng, dropout_key):
                """Evaluate RMSD (not total loss) in inference mode."""
                dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

                def loss_fn(params):
                    logits, updates = state.apply_fn(
                        {"params": params, "batch_stats": state.batch_stats},
                        batch_x, z_rng, train=False,
                        rngs={"dropout": dropout_train_key},
                        mutable=["batch_stats"],
                    )
                    decoded = logits[0]
                    return jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)), (logits, updates)

                return loss_fn(state.params)[0]

        else:
            @jax.jit
            def step(state, batch_x, z_rng, dropout_key):
                """Beta-VAE gradient step without BatchNorm."""
                dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

                def loss_fn(params):
                    logits = state.apply_fn(
                        {"params": params},
                        batch_x, z_rng, train=True,
                        rngs={"dropout": dropout_train_key},
                    )
                    decoded, z_mean, z_logvar = logits
                    rmsd = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)))
                    kl = KL_loss(z_mean, z_logvar)
                    beta = model_ref.beta
                    loss = rmsd + beta * kl
                    return loss, (logits, None)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, (logits, _)), grads = grad_fn(state.params)
                state = state.apply_gradients(grads=grads)
                return state, loss

            @jax.jit
            def evaluate(state, batch_x, z_rng, dropout_key):
                """Evaluate RMSD in inference mode."""
                dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

                def loss_fn(params):
                    logits = state.apply_fn(
                        {"params": params},
                        batch_x, z_rng, train=False,
                        rngs={"dropout": dropout_train_key},
                    )
                    decoded = logits[0]
                    return jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)), (logits, None)

                return loss_fn(state.params)[0]

        return step, evaluate


class VQVAETrainer(Experiment):
    """
    Trainer for VQVAE: RMSD loss + commitment loss.

    The step function minimises:
        log(RMS RMSD) + commitment_weight * commitment_loss

    Parameters
    ----------
    json_fn : str or dict
    make_dirs : bool
    from_json_params : bool
    model_cls : type
        Model class (default VQVAE).
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False, model_cls=VQVAE):
        super().__init__(
            json_fn,
            make_dirs=make_dirs,
            from_json_params=from_json_params,
            model_cls=model_cls,
        )
        commitment_weight = (
            self.json_params.get("vq_commitment_weight", 1.0)
            if hasattr(self, "json_params") else 1.0
        )
        self._step, self._evaluate = self._build_vq_step(commitment_weight)

    def _build_vq_step(self, commitment_weight: float = 1.0):
        """
        Build JIT-compiled step and evaluate closures with commitment loss.

        Parameters
        ----------
        commitment_weight : float
            Weight for the commitment loss (default 1.0).

        Returns
        -------
        step : callable
        evaluate : callable
        """
        atom_rmsd_loss = give_weighted_rmsd_func(jnp.ones(self.model.input_size // 3))
        cw = commitment_weight

        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            """VQ-VAE gradient step."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=True,
                    rngs={"dropout": dropout_train_key},
                )
                decoded, z_e, z_q = logits
                rmsd = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)))
                commitment = jnp.mean((jax.lax.stop_gradient(z_q) - z_e) ** 2)
                codebook_loss = 0.25 * jnp.mean((z_q - jax.lax.stop_gradient(z_e)) ** 2)
                loss = rmsd + cw * (commitment + codebook_loss)
                return loss, (logits, None)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, _)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            """Evaluate RMSD in inference mode."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=False,
                    rngs={"dropout": dropout_train_key},
                )
                decoded = logits[0]
                return jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)), (logits, None)

            return loss_fn(state.params)[0]

        return step, evaluate


class HVAETrainer(Experiment):
    """
    Trainer for HierarchicalVAE: RMSD + KL(z1) + KL(z2).

    The step function minimises:
        log(RMS RMSD) + KL(z1) + KL(z2)

    Parameters
    ----------
    json_fn : str or dict
    make_dirs : bool
    from_json_params : bool
    model_cls : type
        Model class (default HierarchicalVAE).
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False, model_cls=HierarchicalVAE):
        super().__init__(
            json_fn,
            make_dirs=make_dirs,
            from_json_params=from_json_params,
            model_cls=model_cls,
        )
        self._step, self._evaluate = self._build_hvae_step()

    def _build_hvae_step(self):
        """
        Build JIT-compiled step and evaluate closures for the HVAE.

        Returns
        -------
        step : callable
        evaluate : callable
        """
        atom_rmsd_loss = give_weighted_rmsd_func(jnp.ones(self.model.input_size // 3))
        model_ref = self.model

        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            """HVAE gradient step combining RMSD + KL(z1) + KL(z2)."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=True,
                    rngs={"dropout": dropout_train_key},
                )
                decoded, z1_mean, z1_logvar = logits
                rmsd = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)))
                kl1 = KL_loss(z1_mean, z1_logvar)

                # Compute z2 KL via aux_loss on the same params
                # We need to re-run the encoder to get z2 stats
                # Use the model's _encode_full via apply
                rng1, _ = jax.random.split(z_rng)
                z1_m, z1_lv, z2_m, z2_lv = state.apply_fn(
                    {"params": params},
                    batch_x, rng1, train=True,
                    rngs={"dropout": dropout_train_key},
                    method=lambda model, *args, **kwargs: model._encode_full(*args, **kwargs),
                )
                kl2 = KL_loss(z2_m, z2_lv)
                loss = rmsd + kl1 + kl2
                return loss, (logits, None)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, _)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            """Evaluate RMSD in inference mode."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=False,
                    rngs={"dropout": dropout_train_key},
                )
                decoded = logits[0]
                return jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, decoded) ** 2)), (logits, None)

            return loss_fn(state.params)[0]

        return step, evaluate


class FlowTrainer(Experiment):
    """
    Trainer for RealNVPFlow: minimise NLL = 0.5*||z||^2 - log_det.

    RMSD is computed and logged separately (as a diagnostic) but the
    gradient step minimises the negative log-likelihood of the flow.

    Parameters
    ----------
    json_fn : str or dict
    make_dirs : bool
    from_json_params : bool
    model_cls : type
        Model class (default RealNVPFlow).
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False, model_cls=RealNVPFlow):
        super().__init__(
            json_fn,
            make_dirs=make_dirs,
            from_json_params=from_json_params,
            model_cls=model_cls,
        )
        self._step, self._evaluate = self._build_flow_step()

    def _build_flow_step(self):
        """
        Build JIT-compiled step (minimise NLL) and evaluate (log RMSD).

        Returns
        -------
        step : callable
        evaluate : callable
        """
        atom_rmsd_loss = give_weighted_rmsd_func(jnp.ones(self.model.input_size // 3))

        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            """Flow gradient step minimising NLL."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=True,
                    rngs={"dropout": dropout_train_key},
                )
                x_recon, z, _ = logits
                # Compute log_det via aux_loss
                nll = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=True,
                    rngs={"dropout": dropout_train_key},
                    method=lambda model, *a, **kw: model.aux_loss(*a, **kw),
                )
                return nll, (logits, None)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, _)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            """Evaluate reconstruction RMSD (diagnostic, not training loss)."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=False,
                    rngs={"dropout": dropout_train_key},
                )
                x_recon = logits[0]
                return jnp.sqrt(jnp.mean(atom_rmsd_loss(batch_x, x_recon) ** 2)), (logits, None)

            return loss_fn(state.params)[0]

        return step, evaluate


class MAETrainer(Experiment):
    """
    Trainer for MaskedAutoencoder: RMSD loss computed only on masked atoms.

    During training, a random subset of atoms is masked before encoding.
    The loss is the RMSD between predicted and true coordinates for the
    masked atoms only, encouraging the model to reconstruct from context.

    Parameters
    ----------
    json_fn : str or dict
    make_dirs : bool
    from_json_params : bool
    model_cls : type
        Model class (default MaskedAutoencoder).
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False, model_cls=MaskedAutoencoder):
        super().__init__(
            json_fn,
            make_dirs=make_dirs,
            from_json_params=from_json_params,
            model_cls=model_cls,
        )
        self._step, self._evaluate = self._build_mae_step()

    def _build_mae_step(self):
        """
        Build JIT-compiled step and evaluate closures for the MAE.

        Returns
        -------
        step : callable
        evaluate : callable
        """
        n_atoms = self.model.input_size // 3
        mask_ratio = self.model.mask_ratio

        def _masked_rmsd(x_true, x_pred, mask):
            """Per-batch RMSD over masked atoms only."""
            x_t = x_true.reshape(-1, n_atoms, 3)
            x_p = x_pred.reshape(-1, n_atoms, 3)
            diff2 = jnp.sum((x_t - x_p) ** 2, axis=-1)    # (B, N)
            masked_diff2 = jnp.where(mask[None, :], diff2, 0.0)
            n_masked = jnp.sum(mask).clip(min=1)
            per_frame = jnp.sqrt(jnp.sum(masked_diff2, axis=-1) / n_masked)
            return per_frame

        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            """MAE gradient step on masked atoms."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
            mask_rng, z_rng_inner = jax.random.split(z_rng)
            n_keep = max(1, int(round(n_atoms * (1.0 - mask_ratio))))
            perm = jax.random.permutation(mask_rng, n_atoms)
            kept = perm[:n_keep]
            mask = jnp.ones(n_atoms, dtype=bool)
            mask = mask.at[kept].set(False)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng_inner, train=True,
                    rngs={"dropout": dropout_train_key},
                )
                decoded, z_mean, z_logvar = logits
                per_frame = _masked_rmsd(batch_x, decoded, mask)
                loss = jnp.log(jnp.sqrt(jnp.mean(per_frame ** 2)))
                return loss, (logits, None)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, _)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            """Evaluate full reconstruction RMSD in inference mode."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=False,
                    rngs={"dropout": dropout_train_key},
                )
                decoded = logits[0]
                x_t = batch_x.reshape(-1, n_atoms, 3)
                x_p = decoded.reshape(-1, n_atoms, 3)
                per_frame = jnp.sqrt(jnp.mean(jnp.sum((x_t - x_p) ** 2, axis=-1), axis=-1))
                return jnp.sqrt(jnp.mean(per_frame ** 2)), (logits, None)

            return loss_fn(state.params)[0]

        return step, evaluate
