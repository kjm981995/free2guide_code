
## ___***Free2Guide: Gradient-Free Path Integral Control for Enhancing Text-to-Video Generation with Large Vision-Language Models***___

<img width="1521" height="474" alt="image" src="https://github.com/user-attachments/assets/ed358a8e-5cc7-4289-9443-bb620be920a4" />


## ⚙️ Setup

### 1. Install Environment following VideoCrafter
```bash
conda create -n free2guide python=3.8.5
conda activate free2guide
git clone https://github.com/kjm981995/free2guide_code
cd free2guide_code
pip install -r requirements.txt
```

### 2. Download VideoCrafter2 model 

Download pretrained T2V models via [Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt), and put the `model.ckpt` in `checkpoints/base_512_v2/model.ckpt`.

### 3. Inference

Edit OpenAI API key `client = AsyncOpenAI(api_key="YourAPI")` in `path_integral_video.py`. Then run Free2guide code. 

```
python path_integral_video.py
```


## Acknowledgements
Our codebase builds on [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter). 
Thanks the authors for sharing their codebases 

