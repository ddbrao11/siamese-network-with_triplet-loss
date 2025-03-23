# Siamese Network for Face Recognition using Triplet Loss

This repo demonstrates a simple implementation of a Siamese network using Triplet Loss for face recognition in PyTorch.

## 🧠 Key Concepts

- **Siamese Network**: Learns to compare rather than classify.
- **Triplet Loss**: Trains on (anchor, positive, negative) to structure embedding space.

## 📦 Files

- `siamese_network.py`: CNN model to generate embeddings.
- `triplet_loss.py`: Implements triplet loss.
- `train.py`: Shows a dummy training loop.

## 🚀 Getting Started

```bash
pip install torch
python train.py
```

## 🔍 Next Steps

- Replace dummy data with real face images.
- Use triplet mining for better training.
- Add a dataset loader and evaluation metrics.

## 🙌 Author

Made with ❤️ using PyTorch and ChatGPT.