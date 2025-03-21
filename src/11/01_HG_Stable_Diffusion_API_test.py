# 載入套件
import torch
from diffusers import StableDiffusion3Pipeline

# 載入模型
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers"
                                                , torch_dtype=torch.float16)
pipe.to("cuda") # 複製模型到GPU記憶體

# 模型推論，生成圖像
image = pipe(
    prompt="A snake holding a sign that says Happy Lunar Year",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

# 圖像存檔
image.save("sd3_hello_world.png")