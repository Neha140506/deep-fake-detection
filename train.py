from pickle import TRUE
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import torch.optim as optim
from torchvision.transforms import v2
from datasets import load_dataset

print("starting the training")
transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToTensor(),
    v2.Grayscale(1),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])




DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def main():
    # Load datasets

    train_dataset = datasets.ImageFolder(
        root=r"C:\Users\iqbal\Downloads\real_vs_fake\real-vs-fake\train",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers=4)

    val_dataset = datasets.ImageFolder(
        root=r"C:\Users\iqbal\Downloads\real_vs_fake\real-vs-fake\valid",
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,num_workers=4)

    print(f"Number of training samples: {len(train_loader)}")
    print(f"Number of  validation sample: {len(val_loader)}")

    # Load and modify the model
    model = InceptionResnetV1(
        pretrained="vggface2",
        classify=True,
        num_classes=2 
    ).to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    model.logits = torch.nn.Linear(model.logits.in_features, 1)
    model.logits.requires_grad = True  # Train the final layer


    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(2):
        print("Starting training loop")

        model.train()
        running_loss = 0.0
        batch_num=0
        for images, labels in train_loader:
            batch_num+=1
            

            print(f"Processing batch:{batch_num}")
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            model = model.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}")
        # Validation loop (optional)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the fine-tuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'trainedmodel++.pth')

if __name__ == '__main__':
    main()
