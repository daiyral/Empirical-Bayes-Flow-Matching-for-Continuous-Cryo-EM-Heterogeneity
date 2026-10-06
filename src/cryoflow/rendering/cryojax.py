import jax
import jax.numpy as jnp
import cryojax.simulator as cxs
import equinox as eqx
import torch
import numpy as np
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class JaxRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atom_positions, angles, sigma, jax_fn):
        ctx.input_device = atom_positions.device
        pos_np = atom_positions.detach().cpu().numpy()
        angles_np = angles.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        image_jax = jax_fn(pos_np, angles_np, sigma_np)

        ctx.jax_fn = jax_fn
        ctx.pos_np = pos_np
        ctx.angles_np = angles_np
        ctx.sigma_np = sigma_np
        return torch.from_numpy(np.asarray(image_jax))

    @staticmethod
    def backward(ctx, grad_output):
        pos_np = ctx.pos_np
        angles_np = ctx.angles_np
        sigma_np = ctx.sigma_np
        grad_out_np = grad_output.detach().cpu().numpy()

        def jax_forward(pos):
            return ctx.jax_fn(pos, angles_np, sigma_np)

        _, vjp_fn = jax.vjp(jax_forward, pos_np)
        grad_pos_jax, = vjp_fn(grad_out_np)

        grad_pos = torch.from_numpy(np.asarray(grad_pos_jax)).to(ctx.input_device)
        return grad_pos, None, None, None




class JaxRenderBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atom_positions_batch, angles_batch, sigma, jax_fn_batch):
        ctx.input_device = atom_positions_batch.device
        pos_np = atom_positions_batch.detach().cpu().numpy()
        angles_np = angles_batch.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        images_jax = jax_fn_batch(pos_np, angles_np, sigma_np)

        ctx.jax_fn_batch = jax_fn_batch
        ctx.pos_np = pos_np
        ctx.angles_np = angles_np
        ctx.sigma_np = sigma_np
        return torch.from_numpy(np.asarray(images_jax))

    @staticmethod
    def backward(ctx, grad_output):
        pos_np = ctx.pos_np
        angles_np = ctx.angles_np
        sigma_np = ctx.sigma_np
        grad_out_np = grad_output.detach().cpu().numpy()

        def jax_forward(pos):
            return ctx.jax_fn_batch(pos, angles_np, sigma_np)

        _, vjp_fn = jax.vjp(jax_forward, pos_np)
        grad_pos_jax, = vjp_fn(grad_out_np)

        grad_pos = torch.from_numpy(np.asarray(grad_pos_jax)).to(ctx.input_device)
        return grad_pos, None, None, None


class CryoJAXRenderer:
    def __init__(self):
        self.image_size = (64, 64)
        self.pixel_size = 1.2
        self.voltage = 300.0
        self.defocus = 5000
        self.cs = 2.1
        self.amplitude_contrast = 0.06
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._jax_render_fn = self._create_jax_render_fn()
        self._jax_render_fn_noisy = self._create_jax_render_fn(noisy=True)
        self._jax_render_fn_batch = self._create_jax_render_fn_batch()
        # per-particle-defocus variants, built lazily (only used for CTF-error runs)
        self._jax_render_fn_ctf = None
        self._jax_render_fn_batch_ctf = None

    def _create_jax_render_fn(self, noisy=False):
        image_size = self.image_size
        pixel_size = self.pixel_size
        voltage = self.voltage
        defocus = self.defocus
        cs = self.cs
        amplitude_contrast = self.amplitude_contrast

        @jax.jit
        def jax_render(atom_positions, angles, sigma, key=jax.random.PRNGKey(0)):
            bbox_mid = 0.5 * (jnp.min(atom_positions, axis=0, keepdims=True)
                              + jnp.max(atom_positions, axis=0, keepdims=True))
            coords = atom_positions - bbox_mid
            N = coords.shape[0]

            volume = cxs.GaussianMixtureVolume(
                positions=coords,
                variances=jnp.ones(N) * sigma,
                amplitudes=jnp.ones(N),
            )

            image_model = cxs.make_image_model(
                volume=volume,
                normalizes_signal=True,
                image_config=cxs.BasicImageConfig(
                    shape=image_size,
                    pixel_size=pixel_size,
                    voltage_in_kilovolts=voltage,
                ),
                pose=cxs.EulerAnglePose(
                    phi_angle=angles[0],
                    theta_angle=angles[1],
                    psi_angle=angles[2],
                    offset_x_in_angstroms=-0.5 * pixel_size,
                    offset_y_in_angstroms=-0.5 * pixel_size,
                ),
                transfer_theory=cxs.ContrastTransferTheory(
                    ctf=cxs.AstigmaticCTF(
                        defocus_in_angstroms=defocus,
                        spherical_aberration_in_mm=cs,
                    ),
                    amplitude_contrast_ratio=amplitude_contrast,
                ),
            )

            if noisy:
                noise_model = cxs.GaussianWhiteNoiseModel(image_model, variance=0.1)
                image = noise_model.sample(key, outputs_real_space=True)
            else:
                image = image_model.simulate(outputs_real_space=True)
            image = (image - jnp.mean(image)) / (jnp.std(image) + 1e-8)

            return image

        return jax_render

    def _create_jax_render_fn_ctf(self):
        image_size = self.image_size
        pixel_size = self.pixel_size
        voltage = self.voltage
        cs = self.cs
        amplitude_contrast = self.amplitude_contrast

        @jax.jit
        def jax_render(atom_positions, angles, sigma, defocus):
            bbox_mid = 0.5 * (jnp.min(atom_positions, axis=0, keepdims=True)
                              + jnp.max(atom_positions, axis=0, keepdims=True))
            coords = atom_positions - bbox_mid
            N = coords.shape[0]

            image_model = cxs.make_image_model(
                volume=cxs.GaussianMixtureVolume(
                    positions=coords,
                    variances=jnp.ones(N) * sigma,
                    amplitudes=jnp.ones(N),
                ),
                normalizes_signal=True,
                image_config=cxs.BasicImageConfig(
                    shape=image_size,
                    pixel_size=pixel_size,
                    voltage_in_kilovolts=voltage,
                ),
                pose=cxs.EulerAnglePose(
                    phi_angle=angles[0],
                    theta_angle=angles[1],
                    psi_angle=angles[2],
                    offset_x_in_angstroms=-0.5 * pixel_size,
                    offset_y_in_angstroms=-0.5 * pixel_size,
                ),
                transfer_theory=cxs.ContrastTransferTheory(
                    ctf=cxs.AstigmaticCTF(
                        defocus_in_angstroms=defocus,
                        spherical_aberration_in_mm=cs,
                    ),
                    amplitude_contrast_ratio=amplitude_contrast,
                ),
            )
            image = image_model.simulate(outputs_real_space=True)
            return (image - jnp.mean(image)) / (jnp.std(image) + 1e-8)

        return jax_render

    def _create_jax_render_fn_batch_ctf(self):
        image_size = self.image_size
        pixel_size = self.pixel_size
        voltage = self.voltage
        cs = self.cs
        amplitude_contrast = self.amplitude_contrast

        image_config = cxs.BasicImageConfig(
            shape=image_size,
            pixel_size=pixel_size,
            voltage_in_kilovolts=voltage,
        )

        # defocus is vmapped, so the transfer theory is built per-particle and
        # must live in the vmapped partition alongside pose/volume.
        @eqx.filter_vmap(in_axes=(0, 0, 0, None, None), out_axes=(eqx.if_array(0), None))
        def make_models(atom_positions, angles, defocus, sigma, image_config):
            bbox_mid = 0.5 * (jnp.min(atom_positions, axis=0, keepdims=True)
                              + jnp.max(atom_positions, axis=0, keepdims=True))
            coords = atom_positions - bbox_mid
            N = coords.shape[0]
            image_model = cxs.make_image_model(
                volume=cxs.GaussianMixtureVolume(
                    positions=coords,
                    variances=jnp.ones(N) * sigma,
                    amplitudes=jnp.ones(N),
                ),
                normalizes_signal=True,
                image_config=image_config,
                pose=cxs.EulerAnglePose(
                    phi_angle=angles[0],
                    theta_angle=angles[1],
                    psi_angle=angles[2],
                    offset_x_in_angstroms=-0.5 * pixel_size,
                    offset_y_in_angstroms=-0.5 * pixel_size,
                ),
                transfer_theory=cxs.ContrastTransferTheory(
                    ctf=cxs.AstigmaticCTF(
                        defocus_in_angstroms=defocus,
                        spherical_aberration_in_mm=cs,
                    ),
                    amplitude_contrast_ratio=amplitude_contrast,
                ),
            )
            is_var = lambda x: isinstance(
                x, (cxs.AbstractPose, cxs.AbstractAtomVolume, cxs.AbstractTransferTheory)
            )
            filter_spec = jax.tree.map(is_var, image_model, is_leaf=is_var)
            return eqx.partition(image_model, filter_spec)

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
        def simulate_vmap(model_vmap, model_novmap):
            image_model = eqx.combine(model_vmap, model_novmap)
            image = image_model.simulate(outputs_real_space=True)
            return (image - jnp.mean(image)) / (jnp.std(image) + 1e-8)

        @jax.jit
        def jax_render_batch(atom_positions_batch, angles_batch, sigma, defocus_batch):
            model_vmap, model_novmap = make_models(
                atom_positions_batch, angles_batch, defocus_batch, sigma, image_config
            )
            return simulate_vmap(model_vmap, model_novmap)

        return jax_render_batch

    def render_particle_noisy(self, atom_positions, angles, sigma=1.0, seed=0):
        if isinstance(atom_positions, torch.Tensor):
            pos_np = atom_positions.detach().cpu().numpy()
        else:
            pos_np = np.asarray(atom_positions)
        angles_np = np.asarray(angles, dtype=np.float32)
        sigma_np = np.float32(sigma)
        key = jax.random.PRNGKey(seed)
        image_jax = self._jax_render_fn_noisy(pos_np, angles_np, sigma_np, key)
        image = torch.from_numpy(np.asarray(image_jax))
        return image, angles

    def render_particle(self, atom_positions, angles, sigma=1.0, defocus=None):
        original_device = atom_positions.device if isinstance(atom_positions, torch.Tensor) else self.device

        if not isinstance(atom_positions, torch.Tensor):
            atom_positions = torch.tensor(atom_positions, dtype=torch.float32)

        angles_tensor = torch.as_tensor(angles, dtype=torch.float32)
        sigma_tensor = torch.as_tensor(sigma, dtype=torch.float32)

        if defocus is None:
            render_fn = self._jax_render_fn
        else:
            if self._jax_render_fn_ctf is None:
                self._jax_render_fn_ctf = self._create_jax_render_fn_ctf()
            defocus_np = jnp.float32(float(defocus))
            base_fn = self._jax_render_fn_ctf
            render_fn = lambda p, a, s: base_fn(p, a, s, defocus_np)

        image = JaxRenderFunction.apply(atom_positions, angles_tensor, sigma_tensor, render_fn)

        if image.device != original_device:
            image = image.to(original_device)
        return image, angles

    def _create_jax_render_fn_batch(self):
        image_size = self.image_size
        pixel_size = self.pixel_size
        voltage = self.voltage
        defocus = self.defocus
        cs = self.cs
        amplitude_contrast = self.amplitude_contrast

        image_config = cxs.BasicImageConfig(
            shape=image_size,
            pixel_size=pixel_size,
            voltage_in_kilovolts=voltage,
        )
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(
                defocus_in_angstroms=defocus,
                spherical_aberration_in_mm=cs,
            ),
            amplitude_contrast_ratio=amplitude_contrast,
        )

        @eqx.filter_vmap(in_axes=(0, 0, None, None, None), out_axes=(eqx.if_array(0), None))
        def make_models(atom_positions, angles, sigma, image_config, transfer_theory):
            bbox_mid = 0.5 * (jnp.min(atom_positions, axis=0, keepdims=True)
                              + jnp.max(atom_positions, axis=0, keepdims=True))
            coords = atom_positions - bbox_mid
            N = coords.shape[0]
            volume = cxs.GaussianMixtureVolume(
                positions=coords,
                variances=jnp.ones(N) * sigma,
                amplitudes=jnp.ones(N),
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

    def render_particle_batch(self, atom_positions_batch, angles_batch, sigma=1.0, defocus=None):
        original_device = atom_positions_batch.device
        sigma_tensor = torch.as_tensor(sigma, dtype=torch.float32)
        if not isinstance(angles_batch, torch.Tensor):
            angles_batch = torch.as_tensor(angles_batch, dtype=torch.float32)

        if defocus is None:
            render_fn = self._jax_render_fn_batch
        else:
            if self._jax_render_fn_batch_ctf is None:
                self._jax_render_fn_batch_ctf = self._create_jax_render_fn_batch_ctf()
            defocus_np = jnp.asarray(
                defocus.detach().cpu().numpy() if isinstance(defocus, torch.Tensor)
                else np.asarray(defocus), dtype=jnp.float32
            )
            base_fn = self._jax_render_fn_batch_ctf
            render_fn = lambda p, a, s: base_fn(p, a, s, defocus_np)

        images = JaxRenderBatchFunction.apply(
            atom_positions_batch, angles_batch, sigma_tensor, render_fn
        )
        if images.device != original_device:
            images = images.to(original_device)
        return images  # [batch, H, W]
