import torch
from siamese_network import FaceEmbeddingNet
from triplet_loss import TripletLoss

# Dummy training step (replace with real dataset)
def train_step(model, loss_fn, optimizer, anchor, positive, negative):
    model.train()
    anchor_embed = model(anchor)
    positive_embed = model(positive)
    negative_embed = model(negative)

    loss = loss_fn(anchor_embed, positive_embed, negative_embed)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Example usage
if __name__ == "__main__":
    model = FaceEmbeddingNet()
    loss_fn = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Random dummy input (batch_size=4, grayscale 28x28)
    anchor = torch.randn(4, 1, 28, 28)
    positive = torch.randn(4, 1, 28, 28)
    negative = torch.randn(4, 1, 28, 28)

    loss = train_step(model, loss_fn, optimizer, anchor, positive, negative)
    print(f"Training loss: {loss:.4f}")