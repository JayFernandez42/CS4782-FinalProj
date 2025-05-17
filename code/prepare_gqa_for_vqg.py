import json
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import torch
from tqdm import tqdm
import glob
from collections import defaultdict

# ========== CONFIG ==========
project_root = Path(__file__).resolve().parent.parent
IMAGE_DIR = project_root / "data" / "gqa" / "images"
TENSOR_DIR = project_root / "data" / "gqa_resnet_data"
CSV_OUTPUT = project_root / "data" / "gqa_pretrain100k.csv"
WEIGHTS_PATH = project_root / "resnet50-0676ba61.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD RESNET ==========
resnet = models.resnet50()
resnet.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========== LOAD QUESTION JSONs ==========
def load_questions():
    question_paths = sorted(glob.glob(str(project_root / "data/gqa/questions1.2/train_all_questions/*.json")))
    question_paths += [
        project_root / "data/gqa/questions1.2/val_all_questions.json",
        project_root / "data/gqa/questions1.2/test_all_questions.json"
    ]

    image_to_questions = defaultdict(list)
    for path in question_paths:
        with open(path, 'r') as f:
            qjson = json.load(f)
            for q in qjson.values():
                image_to_questions[q["imageId"]].append(q["question"])
    return image_to_questions

# ========== MAIN PROCESSING FUNCTION ==========
def process_gqa_subset(num_unique_images: int):
    image_to_questions = load_questions()
    all_image_ids = list(image_to_questions.keys())
    all_image_ids = [img_id for img_id in all_image_ids if (IMAGE_DIR / f"{img_id}.jpg").exists()]

    processed = 0
    rows = []
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)

    for image_id in tqdm(all_image_ids, desc="Processing images"):
        if processed >= num_unique_images:
            break

        img_path = IMAGE_DIR / f"{image_id}.jpg"
        tensor_fname = f"gqa_{image_id}.pt"
        tensor_path = TENSOR_DIR / tensor_fname

        if not tensor_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = resnet(tensor).squeeze().cpu()
                torch.save(feat, tensor_path)
            except Exception as e:
                print(f"⚠️ Failed to process {image_id}: {e}")
                continue

        # Add all questions for this image
        for question in image_to_questions[image_id]:
            rows.append({
                "tensor_path": tensor_fname,
                "questions": question
            })

        processed += 1

    # Append or create CSV
    df = pd.DataFrame(rows)
    if CSV_OUTPUT.exists():
        df.to_csv(CSV_OUTPUT, mode='a', index=False, header=False)
    else:
        df.to_csv(CSV_OUTPUT, index=False)
    
    print(f"✅ Saved {len(rows)} rows for {processed} unique images to {CSV_OUTPUT}")

process_gqa_subset(100000)
