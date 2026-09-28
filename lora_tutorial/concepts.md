# LoRA Concepts

## 1. Full fine-tuning

The original model parameters are updated.

```text
Base Model
   ↓
Update many/all parameters
   ↓
Fine-tuned Model
```

This can be expensive for large models.

## 2. LoRA
The base model is frozen and small low-rank matrices are trained.

```text
Base Model (frozen)
       +
LoRA Adapter (trainable)
       ↓
Task-specific behavior
```

The mathematical idea is:

```text
W' = W + ΔW
ΔW = B × A
```

`A` and `B` are much smaller than the original weight matrix.

## 3. Important LoRA parameters

### r

The LoRA rank.

Higher `r` can give the adapter more capacity, but increases trainable parameters.

### lora_alpha

Scaling factor for the LoRA update.

### lora_dropout

Dropout applied to LoRA layers during training.

### target_modules

The model layers where LoRA adapters are inserted.

In this tutorial:

```python
target_modules=["q_proj", "v_proj"]
```

These are commonly used attention projection layers.

## 4. Trainable parameter idea

The script prints something similar to:

```text
trainable params: ... || all params: ... || trainable%: ...
```

The important point is that the trainable percentage should be much smaller than 100%.

## 5. LoRA vs QLoRA

LoRA:

```text
Normal/FP model + LoRA
```

QLoRA:

```text
Quantized model + LoRA
```

QLoRA is useful when GPU memory is limited.
