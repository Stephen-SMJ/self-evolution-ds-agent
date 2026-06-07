# GenAI / drawing-with-llms

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 919 quality strong score 59

File: `GenAI/drawing-with-llms/rank919_70pct/fine-tuned-llama-3b.ipynb`
Cells: 7 total, 4 code, 3 markdown

### Cell 0 markdown

```
This notebook implements a submission with Gemma 2 9B IT model with some helper code to ensure the generated SVGs conform to the submission requirements. (See the [Evaluation](https://www.kaggle.com/competitions/drawing-with-llms/overview/evaluation) page for details on the submission requirements.)

To use this notebook interactively, you'll need to install some dependencies. First, *turn on* the Internet under **Session options** to the right. Then select the **Add-ons->Install Dependencies** menu above and click *Run*. A console should pop up with a running `pip` command. Wait for the dependencies to finish installing and then *turn off* the Internet before submitting.
```

### Cell 2 code

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
svg_constraints = kagglehub.package_import('metric/svg-constraints')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bnb_4bit_compute_dtype=torch.float16,
        self.model_path = kagglehub.model_download("felipesens/gemma-3-12b-it-unsloth-bnb-4bit/transformers/4bit-quantized")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.tokenizer.apply_chat_template(
                inputs = self.tokenizer(text=list_of_texts[0], return_tensors="pt").to(DEVICE)
                inputs = self.tokenizer(text=prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                output_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Check and remove invalid href attributes
                        'Removing invalid href attribute in element "%s".', tag_name
            # Validate path elements to help ensure SVG conversion
                # Use regex to validate 'd' attribute format
                    r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                logging.debug('Path element "d" attribute validated (regex check).')
```

### Cell 3 markdown

```
The following code tests the above model in a local mock-up of this competition's evaluation pipeline. It runs the model on a sample of 15 instances defined in the `test.csv` file in the `kaggle_evaluation` package folder.
```

### Cell 5 markdown

```
Alternatively, you could use the code below to run the model over `train.csv` and see some generated images along with some debugging info. Feel free to turn down the logging level to `INFO` if you just want to see the images.
```

### Cell 6 code

```
    train = pl.read_csv('/kaggle/input/drawing-with-llms/train.csv')
```

## 40pct rank 507 quality strong score 47

File: `GenAI/drawing-with-llms/rank507_40pct/lb-0-52-sd-svg-conversion-using-vtracer.ipynb`
Cells: 4 total, 4 code, 0 markdown

### Cell 1 code

```
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
svg_constraints = kagglehub.package_import('metric/svg-constraints')
        self.model_name = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
        self.generator = torch.Generator("cuda:0").manual_seed(1024)
        scheduler = DDIMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        self.model = StableDiffusionPipeline.from_pretrained(self.model_name,scheduler=scheduler,torch_dtype=torch.float16,safety_checker=None)
                    hierarchical="cutout",  # ["stacked"] or "cutout"
                    corner_threshold=60,  # default: 60
                    length_threshold=10,  # in [3.5, 10] default: 4.0
                    splice_threshold=45,  # default: 45
            # Check and remove invalid href attributes
                        'Removing invalid href attribute in element "%s".', tag_name
            # Validate path elements to help ensure SVG conversion
                # Use regex to validate 'd' attribute format
                    r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                logging.debug('Path element "d" attribute validated (regex check).')
```

### Cell 3 code

```
# train = pl.read_csv('/kaggle/input/drawing-with-llms/train.csv')[:10]
```

## 20pct rank 249 quality usable score 35

File: `GenAI/drawing-with-llms/rank249_20pct/getting-old-school-with-gemma-2b-it.ipynb`
Cells: 5 total, 5 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 2 code

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
svg_constraints = kagglehub.package_import('metric/svg-constraints')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = kagglehub.model_download('google/gemma/Transformers/2b-it/3')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            torch_dtype=torch.float16,
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.
                inputs = self.tokenizer(text=prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                output_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Check and remove invalid href attributes
                        'Removing invalid href attribute in element "%s".', tag_name
            # Validate path elements to help ensure SVG conversion
                # Use regex to validate 'd' attribute format
                    r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                logging.debug('Path element "d" attribute validated (regex check).')
```

### Cell 4 code

```
    train = pl.read_csv('/kaggle/input/drawing-with-llms/train.csv')
```

## 10pct rank 123 quality strong score 61

File: `GenAI/drawing-with-llms/rank123_10pct/drawing-with-llms-sdxl-turbo-v2.ipynb`
Cells: 46 total, 37 code, 9 markdown

### Cell 0 markdown

```
add metric optimize: max(orc_score x aes_score)
```

### Cell 2 code

```
import torch
import torch.nn.functional as F
# MODIFIED: Added AutoPipelineForText2Image, StableDiffusionPipeline and EulerDiscreteScheduler might be removed if not used elsewhere.
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoPipelineForText2Image
from transformers import AutoProcessor, AutoModel
metric = kagglehub.package_import('jiazhuang/svg-image-fidelity')
```

### Cell 5 code

```
# print(f"Inspecting metric object: {metric}")
# print(f"Type of metric: {type(metric)}")
# print("Attributes available in metric (excluding private):")
# for attr in dir(metric):
#     print("\n[DEBUG] 'ImageProcessor' IS a direct attribute of metric.")
#     METRIC_IMAGE_PROCESSOR = metric.ImageProcessor # 修正の必要なし
#     print("\n[DEBUG] 'ImageProcessor' IS NOT a direct attribute of metric.")
#     if hasattr(metric, 'utils') and hasattr(metric.utils, 'ImageProcessor'):
#         print("[DEBUG] Found: metric.utils.ImageProcessor")
#         METRIC_IMAGE_PROCESSOR = metric.utils.ImageProcessor # 仮の修正パス
#     elif hasattr(metric, 'helpers') and hasattr(metric.helpers, 'ImageProcessor'):
#         print("[DEBUG] Found: metric.helpers.ImageProcessor")
#         METRIC_IMAGE_PROCESSOR = metric.helpers.ImageProcessor # 仮の修正パス
#     elif hasattr(metric, 'preprocessors') and hasattr(metric.preprocessors, 'ImageProcessor'):
#         print("[DEBUG] Found: metric.preprocessors.ImageProcessor")
#         METRIC_IMAGE_PROCESSOR = metric.preprocessors.ImageProcessor # 仮の修正パス
#         # ここでは、エラーを再現させるために、もし見つからなかった場合は元の metric.ImageProcessor を使うようにしておきます。
#         if 'METRIC_IMAGE_PROCESSOR' not in globals(): # Fallback if not found
#              METRIC_IMAGE_PROCESSOR = metric.ImageProcessor # This will likely raise the original error if not found
```

### Cell 6 code

```
def bitmap_score_instance_impl(multiple_choice_qa, image, random_seed=42):
    image_processor = metric.ImageProcessor(image=image_resize(image), seed=group_seed).apply()
    aesthetic_score = metric.aesthetic_evaluator.score(image)
    vqa_score = metric.vqa_evaluator.score(questions, choices, answers, image)
    ocr_score = metric.vqa_evaluator.ocr(image_processor.image)
    instance_score = metric.harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
    return instance_score, vqa_score, ocr_score, aesthetic_score
def bitmap_score_instance(multiple_choice_qa, image, random_seed=42):
    score_df = []
        instance_score, vqa_score, ocr_score, aesthetic_score = bitmap_score_instance_impl(one_multiple_choice_qa, one_image, random_seed=42)
        results.append(instance_score)
        score_df.append([instance_score, vqa_score, ocr_score, aesthetic_score])
    score_df = pd.DataFrame(score_df, columns=['competition_score', 'vqa_score', 'ocr_score', 'aesthetic_score'])
        return score_df.iloc[0].to_dict()
        return float(fidelity), score_df
```

### Cell 8 code

```
device = "cuda:1" if torch.cuda.is_available() else "cpu" # Note: "cuda:1" implies a multi-GPU setup. For single GPU, "cuda" or "cuda:0" is common.
import torch
model_path = kagglehub.model_download("kawchar85/sdxl-turbo/pytorch/default/2")
    torch_dtype=torch.float16,
```

### Cell 11 code

```
train_df = pd.read_csv(f'{drawing_with_llms_path}/train.csv')
```

### Cell 12 code

```
train_question_df = train_question_df.groupby('id').apply(lambda df: df.to_dict(orient='list'))
train_df = pd.merge(train_df, train_question_df, how='left', on='id')
```

### Cell 19 code

```
bitmap_score_instance(r.multiple_choice_qa, image, random_seed=42)
```

### Cell 22 code

```
def extract_features_by_scale(img_np, num_colors=22, merge_kernel: int = 5):
    order = np.argsort(-counts)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_kernel, merge_kernel))
        header_match = re.match(r"^.*?<svg[^>]*?>", svg, flags=re.S)
```

### Cell 23 code

```
# Import the metric package
metric = kagglehub.package_import('jiazhuang/svg-image-fidelity') # This was already imported, re-import is fine
aesthetic_evaluator = metric.aesthetic_evaluator
svg_to_png = metric.svg_to_png
def high_score_svg_resize(
```

### Cell 26 code

```
metric.score_instance(r.multiple_choice_qa, svg_content, random_seed=42)
```

### Cell 27 code

```
def get_aes_and_ocr_score(svg_content):
        png_image = metric.svg_to_png(svg_content)
        if png_image is None: # Handle case where svg_to_png might return None for invalid SVGs
             return 0.0, 0.0 # Or some other default low score
        return 0.0, 0.0 # Default low score on error
    image_processor = metric.ImageProcessor(image=png_image, seed=33).apply()
    aesthetic_score = metric.aesthetic_evaluator.score(image)
    ocr_score = metric.vqa_evaluator.ocr(image_processor.image)
    return aesthetic_score, ocr_score
```

## 1st rank 1 quality strong score 66

File: `GenAI/drawing-with-llms/rank1_1st/new-team-name-1st-place-solution.ipynb`
Cells: 17 total, 11 code, 6 markdown

### Cell 2 code

```
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
KH_MODEL_DIFF0 = "conormacamhlaoibh/segmind-ssd-1b/transformers/default/1"
KH_MODEL_DIFF1 = "conormacamhlaoibh/segmind-ssd-1b/transformers/default/1"
KH_MODEL_VAE = "conormacamhlaoibh/madebyollin-taesd/transformers/default/2"
KH_MODEL_CLIP = "metric/openai-clip-vit-large-patch14"
KH_MODEL_AEST = "metric/sac-logos-ava1-l14-linearmse"
DEVICE_0 = torch.device("cuda:0")
DEVICE_1 = torch.device("cuda:1")
    "no shading, no blending, no background elements, "
```

### Cell 4 code

```
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    transformers.set_seed(seed)
g0 = torch.Generator(device="cuda:0").manual_seed(SEED)
g1 = torch.Generator(device="cuda:1").manual_seed(SEED + 1)
```

### Cell 8 code

```
    fshift = np.fft.fftshift(f, axes=(0, 1))
    fshift *= mask[:,:,None]
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift, axes=(0, 1)), axes=(0, 1))
    return np.clip(img_back.real, 0, 255).astype(np.uint8)
```

### Cell 11 markdown

```
# Model
```

### Cell 12 code

```
            torch_dtype=torch.float16,
            torch_dtype=torch.float16,
        torch.backends.cuda.matmul.allow_tf32 = True
        model_diff.scheduler = DPMSolverMultistepScheduler.from_config(model_diff.scheduler.config)
        state_dict = torch.load(self.path_model_aest, weights_only=True, map_location=DEVICE_0)
        scores = []
        tensors = [self.proc_clip(image).to(DEVICE_0) for image in images]
        with torch.no_grad():
                x = torch.stack(batch, dim=0)
                scores.extend(preds.cpu().tolist())
        n_unique = len(scores) // n_copies
        scores = np.array(scores).reshape(n_unique, n_copies)
        mean_scores = np.mean(scores, axis=1).tolist()
        best_score, best_score_real = 0.0, 0.0
        for i, _score in enumerate(mean_scores):
            score = _score - penalty
            if score > best_score:
                best_score = score
                best_score_real = _score
        return best_idx, best_score_real
        if self.verbose: print(f"[+] Best mean aesthetic score: {max_aest:.3f}")
        torch.cuda.empty_cache()
```
