# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )



from .resshift_respace import ResShiftSpacedDiffusion, resshift_space_timesteps, ResShiftSpacedDiffusionDDPM

def create_resshift_diffusion(
    *,
    normalize_input,
    schedule_name,
    sf=4,
    min_noise_level=0.01,
    steps=1000,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    weighted_mse=False,
    predict_type='xstart',
    timestep_respacing=None,
    scale_factor=None,
    latent_flag=True,
):
    sqrt_etas = gd.get_named_eta_schedule(
            schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    if predict_type == 'xstart':
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON
    elif predict_type == 'epsilon_scale':
        model_mean_type = gd.ModelMeanType.EPSILON_SCALE
    elif predict_type == 'residual':
        model_mean_type = gd.ModelMeanType.RESIDUAL
    else:
        raise ValueError(f'Unknown Predicted type: {predict_type}')
    return ResShiftSpacedDiffusion(
        use_timesteps=resshift_space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=gd.LossType.WEIGHTED_MSE if weighted_mse else gd.LossType.MSE,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        sf=sf,
        latent_flag=latent_flag,
    )

def create_resshift_diffusion_ddpm(
    *,
    beta_start,
    beta_end,
    sf=4,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    timestep_respacing=None,
    scale_factor=1.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, beta_start, beta_end)
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    return ResShiftpacedDiffusionDDPM(
        use_timesteps=resshift_space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarTypeDDPM.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarTypeDDPM.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarTypeDDPM.LEARNED_RANGE
        ),
        scale_factor=scale_factor,
        sf=sf,
    )