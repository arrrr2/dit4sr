# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math
import torch

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    RESIDUAL = enum.auto()  # the model predicts epsilon
    EPSILON_SCALE = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL
    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    获取预定义的 beta 调度器。

    支持 'linear' 和 'squaredcos_cap_v2' 两种调度器。

    Args:
        schedule_name (str): 调度器名称，支持 'linear' 和 'squaredcos_cap_v2'。
        num_diffusion_timesteps (int): 扩散步骤数。
        beta_start (float, optional): 起始 beta 值，仅对 'linear' 有效。默认值为 0.0001。
        beta_end (float, optional): 结束 beta 值，仅对 'linear' 有效。默认值为 0.02。

    Returns:
        np.ndarray: 生成的 beta 调度序列。
    """
    if schedule_name == "linear":
        # 线性调度，从 beta_start 线性增长到 beta_end
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    elif schedule_name == "resshift_linear":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2


    elif schedule_name == "squaredcos_cap_v2":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    
    assert betas.shape == (num_diffusion_timesteps,), "Beta schedule shape mismatch."
    return betas

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        # etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas

class GaussianDiffusion:

    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, lq=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if self.model_mean_type == ModelMeanType.EPSILON:
                terms["model_outputs"]  = self._predict_xstart_from_eps(x_t, t, model_output)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
        

    def _gaussian_weights(self, tile_width, tile_height, nbatches, input_channels):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.05
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        weights_tensor = th.tensor(weights, device="cpu")
        return th.tile(weights_tensor, (nbatches, input_channels, 1, 1)) # Assuming model_channels is accessible, otherwise replace with C


    def get_patches(self, x, patch_size, stride):
        """
        将图像分割成重叠的小块。

        Args:
            x: 待分割的图像张量，形状为 (N, C, H, W)。
            patch_size: 小块的大小 (假设小块是正方形的)。
            stride: 步长。

        Returns:
            patches: 分割后的小块张量，形状为 (num_patches, C, patch_size, patch_size)。
            patch_coords: 每个小块的坐标列表，每个坐标是一个元组 (row_start, col_start)。
        """
        n, c, h, w = x.shape
        assert n == 1, "Batch size must be 1 for patch aggregation."

        patches = []
        patch_coords = []

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = x[:, :, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                patch_coords.append((i, j))

        patches = th.cat(patches, dim=0)

        return patches, patch_coords

    def aggregate_patches(self, patches, patch_coords, full_size, patch_size, tile_weights):
        """
        将小块聚合回完整图像。

        Args:
            patches: 小块张量，形状为 (num_patches, C, patch_size, patch_size)。
            patch_coords: 每个小块的坐标列表，每个坐标是一个元组 (row_start, col_start)。
            full_size: 完整图像的大小 (例如，(height, width))。
            patch_size: 小块的大小。
            tile_weights: 高斯权重矩阵

        Returns:
            聚合后的图像，形状为 (1, C, full_size[0], full_size[1])。
        """
        num_patches = patches.shape[0]
        channels = patches.shape[1]
        aggregated_image = th.zeros((1, channels, full_size[0], full_size[1]), device=patches.device)
        contributors = th.zeros((1, channels, full_size[0], full_size[1]), device=patches.device) # Changed to channels dim

        for i in range(num_patches):
            row_start, col_start = patch_coords[i]

            # 将加权后的小块累加到聚合图像
            aggregated_image[:, :, row_start:row_start+patch_size, col_start:col_start+patch_size] += patches[i].unsqueeze(0) * tile_weights
            contributors[:, :, row_start:row_start+patch_size, col_start:col_start+patch_size] += tile_weights # Accumulate weights

        # 除以权重总和进行归一化，添加 epsilon 防止除零
        epsilon = 1e-8
        aggregated_image /= (contributors + epsilon) # Divide by contributor sum

        return aggregated_image


    def p_sample_loop_progressive_with_patch_aggregation(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        patch_size=64,
        stride=32,
        batch_size_patch=4
    ):
        """
        使用块聚合生成样本并生成每个时间步的中间样本。

        参数与 p_sample_loop_with_patch_aggregation() 相同。

        返回一个字典生成器，其中每个字典是 p_sample_with_patch_aggregation() 的返回值。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        full_size = (shape[2], shape[3])

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        tile_weights = self._gaussian_weights(patch_size, patch_size, 1) # Calculate tile_weights once

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                img = self.p_sample_with_patch_aggregation(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    patch_size=patch_size,
                    stride=stride,
                    tile_weights=tile_weights,
                    batch_size_patch=batch_size_patch
                )
                yield {"sample": img, "pred_xstart": img} # In patch aggregation, sample and pred_xstart are the same aggregated image


    def p_sample_with_patch_aggregation(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        patch_size=64,
        stride=32,
        tile_weights=None,
        batch_size_patch=4
    ):
        """
        使用块聚合从给定时间步的模型中采样 x_{t-1}。

        Args:
            model: 要从中采样的模型。
            x: 当前的张量 x_t，形状为 (N, C, H, W)。
            t: t 的值，对于第一个扩散步骤从 0 开始。
            clip_denoised: 如果为 True，则将 x_start 预测裁剪到 [-1, 1]。
            denoised_fn: 如果不为 None，则在采样之前将此函数应用于 x_start 预测。
            cond_fn: 如果不为 None，则这是一个梯度函数，其作用类似于模型。
            model_kwargs: 如果不为 None，则将额外的关键字参数传递给模型。这可以用于条件控制。
            patch_size: 小块的大小。
            stride: 步长.
            tile_weights: 高斯权重矩阵.
            batch_size_patch: 每个batch处理的patch数量.

        Returns:
            聚合后的去噪图像，形状为 (1, C, H, W)
        """
        assert tile_weights is not None

        b, c, h, w = x.shape # b should be 1 here.

        grid_rows = 0
        cur_x = 0
        while cur_x < w:
            cur_x = max(grid_rows * patch_size - stride * grid_rows, 0) + patch_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < h:
            cur_y = max(grid_cols * patch_size - stride * grid_cols, 0) + patch_size
            grid_cols += 1

        input_patches = []
        patch_coords = []
        noise_preds = []

        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    # extract tile from input image
                    ofs_x = max(row * patch_size - stride * row, 0)
                    ofs_y = max(col * patch_size - stride * col, 0)
                    # input tile area on total image
                if row == grid_rows - 1:
                    ofs_x = w - patch_size
                if col == grid_cols - 1:
                    ofs_y = h - patch_size

                input_start_x = ofs_x
                input_end_x = ofs_x + patch_size
                input_start_y = ofs_y
                input_end_y = ofs_y + patch_size

                # input tile dimensions
                input_tile = x[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_patches.append(input_tile)
                patch_coords.append((input_start_y, input_start_x))

                if len(input_patches) == batch_size_patch or (row == grid_rows - 1 and col == grid_cols - 1):
                    input_patches_tensor = th.cat(input_patches, dim=0)
                    t_patches = t.repeat(input_patches_tensor.size(0)) # Repeat time tensor for batch of patches

                    model_kwargs_patches = model_kwargs.copy()
                    if 'y' in model_kwargs_patches and model_kwargs_patches['y'] is not None:
                        latent_patches, _ = self.get_patches(model_kwargs_patches['y'], patch_size, stride)
                        model_kwargs_patches['y'] = latent_patches[:input_patches_tensor.size(0)] # Assuming latent patches are pre-calculated and aligned

                    out_patches = self.p_mean_variance(
                        model,
                        input_patches_tensor,
                        t_patches,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs_patches,
                    )
                    noise_patches = th.randn_like(input_patches_tensor)
                    nonzero_mask = (
                        (t_patches != 0).float().view(-1, *([1] * (len(input_patches_tensor.shape) - 1)))
                    )
                    if cond_fn is not None:
                        out_patches["mean"] = self.condition_mean(cond_fn, out_patches, input_patches_tensor, t_patches, model_kwargs=model_kwargs_patches)
                    sample_patches = out_patches["mean"] + nonzero_mask * th.exp(0.5 * out_patches["log_variance"]) * noise_patches

                    for sample_i in range(sample_patches.size(0)):
                        noise_preds_row.append(sample_patches[sample_i].unsqueeze(0))

                    input_patches = [] # Clear for next batch
                    patch_coords = [] # Clear for next batch
            noise_preds.append(noise_preds_row)


        # Stitch noise predictions for all tiles
        noise_pred = th.zeros(x.shape, device=x.device)
        contributors = th.zeros(x.shape, device=x.device)
        patch_index = 0
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    # extract tile from input image
                    ofs_x = max(row * patch_size - stride * row, 0)
                    ofs_y = max(col * patch_size - stride * col, 0)
                    # input tile area on total image
                if row == grid_rows - 1:
                    ofs_x = w - patch_size
                if col == grid_cols - 1:
                    ofs_y = h - patch_size

                input_start_x = ofs_x
                input_end_x = ofs_x + patch_size
                input_start_y = ofs_y
                input_end_y = ofs_y + patch_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row][col] * tile_weights # Apply weights and accumulate
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights # Accumulate weights

        # Average overlapping areas with more than 1 contributor
        noise_pred /= (contributors + 1e-8) # Avoid division by zero
        return noise_pred


    def ddim_sample_loop_progressive_with_patch_aggregation(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        patch_size=64,
        stride=32,
        batch_size_patch=4
    ):
        """
        使用块聚合和 DDIM 生成样本并生成每个时间步的中间样本。

        参数与 ddim_sample_loop_with_patch_aggregation() 相同。

        返回一个字典生成器，其中每个字典是 ddim_sample_with_patch_aggregation() 的返回值。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        full_size = (shape[2], shape[3])

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        tile_weights = self._gaussian_weights(patch_size, patch_size, 1, shape[1] // 2) # Calculate tile_weights once

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                img = self.ddim_sample_with_patch_aggregation(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    patch_size=patch_size,
                    stride=stride,
                    tile_weights=tile_weights,
                    batch_size_patch=batch_size_patch
                )
                yield {"sample": img, "pred_xstart": img} # In patch aggregation, sample and pred_xstart are the same aggregated image


    def ddim_sample_with_patch_aggregation(
            self,
            model,
            x,  # 输入张量，形状 [1, C, H, W]
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
            patch_size=64,
            stride=32,
            tile_weights=None,
            batch_size_patch=4
        ):
            assert tile_weights is not None
            model_kwargs = model_kwargs or {}
            device = x.device

            # ===================================================================
            # 1. 预计算所有块的坐标 (使用 left/top 避免命名冲突)
            # ===================================================================
            def compute_positions(total_len):
                positions = []
                pos = 0
                while pos + patch_size <= total_len:
                    positions.append(pos)
                    pos += (patch_size - stride) 
                # 处理最后一个块（确保对齐边界）
                if positions[-1] + patch_size < total_len:
                    positions.append(total_len - patch_size)
                return positions

            # 计算水平和垂直坐标
            h, w = x.shape[2], x.shape[3]
            x_positions = compute_positions(w)  # left 坐标列表
            y_positions = compute_positions(h)  # top 坐标列表

            # 生成所有坐标对 (top, left)
            all_coords = [(top, left) for top in y_positions for left in x_positions]
            

            # ===================================================================
            # 2. 分批次处理块
            # ===================================================================
            all_processed_patches = []
            
            for batch_start in range(0, len(all_coords), batch_size_patch):
                batch_coords = all_coords[batch_start:batch_start+batch_size_patch]
                # print(batch_coords)
                
                # ---------------------------------------------------------------
                # 2.1 提取当前批次的x和y块
                # ---------------------------------------------------------------
                x_batch, y_batch = [], []
                
                for (top, left) in batch_coords:  # 关键修改：使用 top/left 命名
                    # 提取x的块
                    x_patch = x[:, :, top:top+patch_size, left:left+patch_size]
                    x_batch.append(x_patch)
                    
                    # 提取条件y的块（如果存在）
                    if 'y' in model_kwargs and model_kwargs['y'] is not None:
                        y_patch = model_kwargs['y'][:, :, top:top+patch_size, left:left+patch_size]
                        y_batch.append(y_patch)
                
                # 构造批次输入
                x_tensor = th.cat(x_batch, dim=0)
                t_tensor = t.expand(x_tensor.size(0))
                model_kwargs_batch = model_kwargs.copy()
                if y_batch:
                    model_kwargs_batch['y'] = th.cat(y_batch, dim=0)
                
                # ---------------------------------------------------------------
                # 2.2 执行采样
                # ---------------------------------------------------------------
                with th.no_grad():
                    out = self.ddim_sample(
                        model,
                        x_tensor,
                        t_tensor,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs_batch,
                        eta=eta
                    )
                    all_processed_patches.extend(out["sample"].unbind(dim=0))

            # ===================================================================
            # 3. 融合所有处理后的块
            # ===================================================================
            final_output = th.zeros_like(x)
            weight_map = th.zeros_like(x)
            tile_weights = tile_weights.to(device)

            for (top, left), patch in zip(all_coords, all_processed_patches):  # 关键修改：使用 top/left
                # 计算实际写入区域
                h_start, w_start = top, left
                h_end = min(h_start + patch_size, h)
                w_end = min(w_start + patch_size, w)
                
                # 获取有效区域
                # valid_patch = patch[None, :, :h_end-h_start, :w_end-w_start]
                # valid_weights = tile_weights[:, :, :h_end-h_start, :w_end-w_start]
                
                # 累加结果
                # final_output[:, :, h_start:h_end, w_start:w_end] += valid_patch * valid_weights
                # weight_map[:, :, h_start:h_end, w_start:w_end] += valid_weights

                final_output [:, :, h_start:h_end, w_start:w_end] += patch * tile_weights
                weight_map[:, :, h_start:h_end, w_start:w_end] += tile_weights
            # 归一化处理
            return final_output / weight_map.clamp(min=1e-8)
    
class ResShiftDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor
    """

    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type,
        loss_type,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # weight for the mse loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]  :
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                    kappa * self.alpha / ((1-self.etas) * self.sqrt_etas)
                    )**2
        else:
            raise NotImplementedError(model_mean_type)

        # self.weight_loss_mse = np.append(weight_loss_mse[1],  weight_loss_mse[1:])
        self.weight_loss_mse = weight_loss_mse

    def q_mean_variance(self, x_start, y, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(self, x_start, y, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:      # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)

    def p_sample(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}

    def p_sample_loop(
        self,
        y,
        model,
        first_stage_model=None,
        consistencydecoder=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param model: the model module.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            y,
            model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample["sample"]
        with th.no_grad():
            out = self.decode_first_stage(
                    final,
                    first_stage_model=first_stage_model,
                    consistencydecoder=consistencydecoder,
                    )
        return out

    def p_sample_loop_progressive(
            self, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

        # generating noise
        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                yield out
                z_sample = out["sample"]

    def decode_first_stage(self, z_sample, first_stage_model=None, consistencydecoder=None):
        batch_size = z_sample.shape[0]
        data_dtype = z_sample.dtype

        if consistencydecoder is None:
            model = first_stage_model
            decoder = first_stage_model.decode
            model_dtype = next(model.parameters()).dtype
        else:
            model = consistencydecoder
            decoder = consistencydecoder
            model_dtype = next(model.ckpt.parameters()).dtype

        if first_stage_model is None:
            return z_sample
        else:
            z_sample = 1 / self.scale_factor * z_sample
            if consistencydecoder is None:
                out = decoder(z_sample.type(model_dtype))
            else:
                with th.cuda.amp.autocast():
                    out = decoder(z_sample)
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        data_dtype = y.dtype
        model_dtype = next(first_stage_model.model.parameters()).dtype
        if up_sample and self.sf != 1:
            y = torch.nn.functional.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            if not model_dtype == data_dtype:
                y = y.type(model_dtype)
            with th.no_grad():
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = th.randn_like(y)

        t = th.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()

        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def training_losses(
            self, model, x_start, y, t,
            first_stage_model=None,
            model_kwargs=None,
            noise=None,
            ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param first_stage_model: autoencoder model
        :param x_start: the [N x C x ...] tensor of inputs. HQ images.
        :param y: the [N x C x ...] tensor of degraded inputs. LQ images.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param up_sample_lq: Upsampling low-quality image before encoding
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        if noise is None:
            noise = th.randn_like(z_start)

        z_t = self.q_sample(z_start, z_y, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.WEIGHTED_MSE:
            model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
            target = {
                ModelMeanType.START_X: z_start,
                ModelMeanType.RESIDUAL: z_y - z_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == z_start.shape, f"model_output.shape {model_output.shape}, target.shape {target.shape}, z_start.shape {z_start.shape} should be the same"
            terms["loss"] = mean_flat((target - model_output) ** 2)
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms["loss"] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1
            terms["loss"] *= weights
        else:
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output)
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output)
        else:
            raise NotImplementedError(self.model_mean_type)

        terms["pred_zstart"] = pred_zstart
        terms["z_t"] = z_t

        return terms

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                # the variance of latent code is around 1.0
                std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

class ResShiftDiffusionDDPM:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarTypeDDPM determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        scale_factor=None,
        sf=4,
    ):
        self.model_mean_type = model_mean_type        # EPSILON
        self.model_var_type = model_var_type          # LEARNED_RANGE
        self.scale_factor = scale_factor   # scale factor in latent space default True
        self.sf=sf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:       # predict x_{t-1}
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )                                                  #  predict \eps
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        first_stage_model=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        first_stage_model=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device).long()
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, first_stage_model=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_start = self.encode_first_stage(x_start, first_stage_model)
        if noise is None:
            noise = th.randn_like(z_start)
        z_t = self.q_sample(z_start, t, noise=noise)

        terms = {}

        model_output = model(z_t, t, **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=z_start, x_t=z_t, t=t
            )[0],
            ModelMeanType.START_X: z_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == z_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)   # q(x_t|x_0)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)


    def _scale_input(self, inputs, t):
        return inputs

    def decode_first_stage(self, z_sample, first_stage_model=None):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        else:
            with th.no_grad():
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample)
                return out.type(ori_dtype)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = torch.nn.functional.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            with th.no_grad():
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
                return out.type(ori_dtype)

