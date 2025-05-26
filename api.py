import os
import random
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

# === CONFIG ===
MODEL_PATH = "models/BAGEL-7B-MoT"

# === MODEL LOADING ===
llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(MODEL_PATH, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

device_map = infer_auto_device_map(
    model,
    max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = first_device
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        device_map[k] = first_device

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(MODEL_PATH, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    offload_folder="offload",
    dtype=torch.bfloat16,
    force_hooks=True,
).eval()

inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

# === UTILITY ===
def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# === FASTAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# === ENDPOINTS ===

@app.post("/text-to-image/")
async def text_to_image_api(
    prompt: str = Form(...),
    show_thinking: bool = Form(False),
    image_ratio: str = Form("1:1"),
    cfg_text_scale: float = Form(4.0),
    cfg_interval: float = Form(0.4),
    timestep_shift: float = Form(3.0),
    num_timesteps: int = Form(50),
    cfg_renorm_min: float = Form(1.0),
    cfg_renorm_type: str = Form("global"),
    max_think_token_n: int = Form(1024),
    do_sample: bool = Form(False),
    text_temperature: float = Form(0.3),
    seed: int = Form(0)
):
    set_seed(seed)
    shape_map = {
        "1:1": (1024, 1024),
        "4:3": (768, 1024),
        "3:4": (1024, 768),
        "16:9": (576, 1024),
        "9:16": (1024, 576),
    }
    image_shapes = shape_map.get(image_ratio, (1024, 1024))

    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )

    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    buffer = BytesIO()
    result["image"].save(buffer, format="PNG")
    return JSONResponse(content={"text": result.get("text", None)})
    

@app.post("/image-understanding/")
async def image_understanding_api(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    show_thinking: bool = Form(False),
    do_sample: bool = Form(False),
    text_temperature: float = Form(0.3),
    max_new_tokens: int = Form(512)
):
    image = Image.open(BytesIO(await file.read()))
    image = pil_img2rgb(image)

    result = inferencer(
        image=image, text=prompt, think=show_thinking, understanding_output=True,
        do_sample=do_sample, text_temperature=text_temperature, max_think_token_n=max_new_tokens
    )
    return JSONResponse(content={"text": result["text"]})


@app.post("/edit-image/")
async def edit_image_api(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    show_thinking: bool = Form(False),
    cfg_text_scale: float = Form(4.0),
    cfg_img_scale: float = Form(2.0),
    cfg_interval: float = Form(0.0),
    timestep_shift: float = Form(3.0),
    num_timesteps: int = Form(50),
    cfg_renorm_min: float = Form(1.0),
    cfg_renorm_type: str = Form("text_channel"),
    max_think_token_n: int = Form(1024),
    do_sample: bool = Form(False),
    text_temperature: float = Form(0.3),
    seed: int = Form(0)
):
    set_seed(seed)
    image = Image.open(BytesIO(await file.read()))
    image = pil_img2rgb(image)

    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )

    result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)
    buffer = BytesIO()
    result["image"].save(buffer, format="PNG")
    return JSONResponse(content={"text": result.get("text", "")})
