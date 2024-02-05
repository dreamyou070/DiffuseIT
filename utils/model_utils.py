from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults
def get_model_config(args):
    model_config = model_and_diffusion_defaults()
    model_config.update({"attention_resolutions": "32, 16, 8",
                         "class_cond": args.model_output_size == 512,  # class_cond = False
                         "diffusion_steps": 1000,
                         "rescale_timesteps": True,
                         "timestep_respacing": args.timestep_respacing,
                         "image_size": args.model_output_size,  # 256
                         "learn_sigma": True,
                         "noise_schedule": "linear",
                         "num_channels": 256,
                         "num_head_channels": 64,
                         "num_res_blocks": 2,
                         "resblock_updown": True,
                         "use_fp16": True,
                         "use_scale_shift_norm": True, })
    return model_config

def unscale_timestep(diffusion, t):
    unscaled_timestep = (t * (diffusion.num_timesteps / 1000)).long()
    return unscaled_timestep