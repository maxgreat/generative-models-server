from flask import Flask, request, send_file
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from omegaconf import OmegaConf
from einops import rearrange, repeat
from sgm.util import exists, instantiate_from_config
import cv2
import math


app = Flask(__name__)

device: str = 'cuda'
num_frames: int = 14
num_steps: int = 25
fps_id: int = 6
motion_bucket_id: int = 127
cond_aug: float = 0.02
seed: int = 23
decoding_t: int = 14
config: str = "scripts/sampling/configs/svd.yaml"

config = OmegaConf.load(config)
if device == "cuda":
    config.model.params.conditioner_config.params.emb_models[
        0
    ].params.open_clip_embedding_config.params.init_device = device

config.model.params.sampler_config.params.num_steps = num_steps
config.model.params.sampler_config.params.guider_config.params.num_frames = (
    num_frames
)
if device == "cuda":
    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval()
else:
    model = instantiate_from_config(config.model).to(device).eval()



@app.route('/video_diffusion', methods=['POST'])
def video_diffusion():
    # Check if an image is provided
    if 'image' not in request.files:
        return "No image provided", 400

    file = request.files['image']
    if file:
        # Convert the image file to an Image object
        image = Image.open(BytesIO(file.read())).convert('RGB')
        w, h = image.size
        if h % 64 != 0 or w % 64 != 0: #resize image to a %64
            width, height = map(lambda x: x - x % 64, (w, h))
            image = image.resize((width, height))
        image = ToTensor()(image)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                video_path = f"temp.mp4" # TODO change to request ID
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    fps_id + 1,
                    (samples.shape[-1], samples.shape[-2]),
                )
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                for frame in vid:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)
                writer.release()

        # Send the video file back
        return send_file(video_path, mimetype='video/mp4')

    return "Invalid request", 400

def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


if __name__ == '__main__':
    app.run(debug=True)