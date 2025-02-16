
# **Multilingual Chatbot Arena – WSDM Cup 2025**  

This project was developed for the **WSDM Cup Multilingual Chatbot Arena** competition on Kaggle. The task involves **ranking chatbot responses** by training models to compare two responses for a given prompt.  

---

## **Project Overview**  

The objective is to **train models that can determine the superior chatbot response** given a prompt. The dataset contains prompts and two corresponding chatbot-generated responses (`response_a` and `response_b`). The model predicts which response is better based on ranking.  

**Techniques & Models Used:**  
- **Pretrained Transformer Models**: Gemma, DeBERTa, and others.  
- **Fine-Tuning with LoRA / QLoRA** for efficient adaptation.  
- **Tokenization Strategies**: Standard and specialized techniques like MCQ-style tokenization.  
- **Multi-GPU Utilization**: Optimized execution across different GPUs.  
- **Ensembling & TTA (Test Time Augmentation)** for robust predictions.  
- **Efficient Parallel Inference with Multi-GPU Execution.**  

There are multiple training scripts, each implementing similar logic but using different models like **Gemma, DeBERTa**, etc. Additionally, helper notebooks may be included, which are not original content.

---

## **Training**  

Each model is trained separately with **LoRA-based fine-tuning** for efficiency. The core training script includes:  
- **Dataset Processing** (prompt-response structuring, tokenization).  
- **LoRA / QLoRA-based Fine-Tuning** for adapting large-scale models efficiently.  
- **Multi-GPU Training** using parallelization.  
- **Optimizer & Scheduler Adjustments** (using AdamW, cosine decay, etc.).  

Training runs require **accelerate, peft, transformers, and bitsandbytes** for efficient execution.  

### **Install Dependencies:**  
```bash
pip install transformers peft accelerate bitsandbytes datasets
```

### **Run Training:**  
Modify and execute the appropriate training script (e.g., for Gemma):  
```python
python training_gemma.py
```

---

## **GPU Utilization & Experiments**  

This project actively **leverages multiple GPUs** on both **Kaggle and Colab Pro** for efficient model execution.  

### **Multi-GPU Execution Strategies:**  
- **GPU Allocation:** Models are distributed across **A100, T4, and V100 GPUs** depending on availability.  
- **Parallelized Inference:** Uses **device_map** in `torch` to distribute workloads across multiple GPUs.  
- **Efficient Tokenization:** Longer sequences are split and dynamically padded to optimize **memory utilization**.  
- **Performance Benchmarking:** Various batch sizes and mixed-precision training (`torch.cuda.amp`) were tested to optimize speed vs. memory constraints.  

### **Colab vs. Kaggle GPU Performance Observations:**  
- **Kaggle A100**: Ideal for **full fine-tuning** and **multi-GPU execution** due to larger VRAM.  
- **Colab T4/V100**: Used for **LoRA-based fine-tuning**, allowing training even on smaller VRAM.  
- **Gradient Checkpointing**: Enabled for models running on lower VRAM GPUs to optimize memory usage.  

---

## **🛠️ Inference**  

The inference pipeline evaluates chatbot responses by ranking the probability of each response being preferred. Key steps:  
- **Load fine-tuned model & tokenizer.**  
- **Tokenize responses efficiently.**  
- **Parallelized Inference on Multi-GPU Setup.**  
- **TTA (Test Time Augmentation)** for improved stability.  

### **Run Inference:**  
```python
python inference.py
```

**Output:** A CSV file containing predictions (`submission.csv`).  

---

## **Additional Notes**  

- MCQ-style tokenization was explored in different contexts but **not used here** explicitly. Although it definitely boosted performance, it did not make a lot of difference in bigger model, and just increased training time, which could be avoided by better tokenization techniques probably.
- Some helper learning notebooks are included but are **not original content**, its just useful for future references.  


---
