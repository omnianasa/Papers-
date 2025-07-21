import pygame
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog

model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)
model.load_state_dict(torch.load("vgg_cats_dogs.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

class_names = ["Dog", "Cat"] 

def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    intensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(intensor)
        _, pred = torch.max(output, 1)
    return image, class_names[pred.item()]

#gui
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("VGG cats and dogs classifier")
font = pygame.font.SysFont(None, 40)
clock = pygame.time.Clock()

img_surface = None
prediction = ""
running = True

while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                tk.Tk().withdraw()
                path = filedialog.askopenfilename()
                if path:
                    pil_img, prediction = predict(path)
                    pil_img = pil_img.resize((300, 300))
                    img_surface = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)

    if img_surface:
        screen.blit(img_surface, (50, 150))
        
    pred_text = font.render(f"Prediction: {prediction}", True, (255, 255, 255))
    screen.blit(pred_text, (50, 50))

    instruct = font.render("Press SPACE to choose image", True, (200, 200, 200))
    screen.blit(instruct, (50, 500))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
