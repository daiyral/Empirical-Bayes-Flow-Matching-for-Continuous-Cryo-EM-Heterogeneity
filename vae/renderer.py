import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cryoflow.rendering.cryojax import CryoJAXRenderer


class GMMRenderer(CryoJAXRenderer):
    # The VAE baseline was trained with K=5 Gaussians per atom (same position,
    # variances = sigma * multipliers) instead of cryoflow's single Gaussian.
    variance_multipliers = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)

    def _create_jax_render_fn_batch(self):
        pixel_size = self.pixel_size
        variance_multipliers = jnp.asarray(self.variance_multipliers)
        K = len(self.variance_multipliers)

        image_config = cxs.BasicImageConfig(
            shape=self.image_size,
            pixel_size=pixel_size,
            voltage_in_kilovolts=self.voltage,
        )
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(
                defocus_in_angstroms=self.defocus,
                spherical_aberration_in_mm=self.cs,
            ),
            amplitude_contrast_ratio=self.amplitude_contrast,
        )

        @eqx.filter_vmap(in_axes=(0, 0, None, None, None), out_axes=(eqx.if_array(0), None))
        def make_models(atom_positions, angles, sigma, image_config, transfer_theory):
            bbox_mid = 0.5 * (jnp.min(atom_positions, axis=0, keepdims=True)
                              + jnp.max(atom_positions, axis=0, keepdims=True))
            coords = atom_positions - bbox_mid
            N = coords.shape[0]

            volume = cxs.GaussianMixtureVolume(
                positions=jnp.repeat(coords, K, axis=0),
                variances=jnp.tile(variance_multipliers * sigma, N),
                amplitudes=jnp.ones(N * K) / K,
            )
            image_model = cxs.make_image_model(
                volume=volume,
                normalizes_signal=True,
                image_config=image_config,
                pose=cxs.EulerAnglePose(
                    phi_angle=angles[0],
                    theta_angle=angles[1],
                    psi_angle=angles[2],
                    offset_x_in_angstroms=-0.5 * pixel_size,
                    offset_y_in_angstroms=-0.5 * pixel_size,
                ),
                transfer_theory=transfer_theory,
            )
            is_var = lambda x: isinstance(x, (cxs.AbstractPose, cxs.AbstractAtomVolume))
            filter_spec = jax.tree.map(is_var, image_model, is_leaf=is_var)
            return eqx.partition(image_model, filter_spec)

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
        def simulate_vmap(model_vmap, model_novmap):
            image_model = eqx.combine(model_vmap, model_novmap)
            image = image_model.simulate(outputs_real_space=True)
            return (image - jnp.mean(image)) / (jnp.std(image) + 1e-8)

        @jax.jit
        def jax_render_batch(atom_positions_batch, angles_batch, sigma):
            model_vmap, model_novmap = make_models(
                atom_positions_batch, angles_batch, sigma, image_config, transfer_theory
            )
            return simulate_vmap(model_vmap, model_novmap)

        return jax_render_batch
