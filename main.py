from fastapi import FastAPI, UploadFile

from models.api import remove_watermark

app = FastAPI()


@app.post("/")
async def main(image_path: UploadFile, mask_path: UploadFile, max_dim: int, reg_noise: float, input_depth: int, lr: float,
               show_step: int, training_steps: int, tqdm_length: int):
    watermarked_extension = image_path.filename.split('.')[-1]
    mask_extension = mask_path.filename.split('.')[-1]
    with open(f"watermarked.{watermarked_extension}", "wb") as f:
        f.write(await image_path.read())
        f.close()
    with open(f"mask.{mask_extension}", "wb") as f:
        f.write(await mask_path.read())
        f.close()
    watermarked_image = remove_watermark(
        f"watermarked.{watermarked_extension}",
        f"mask.{mask_extension}",
        max_dim,
        reg_noise,
        input_depth,
        lr,
        show_step,
        training_steps,
        tqdm_length
    )
    return {"message": "success", "image_path": watermarked_image}


@app.get("/")
async def main():
    return {"message": "success"}

