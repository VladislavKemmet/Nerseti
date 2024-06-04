import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random

# Определение размеров входных и выходных данных
input_size = 28 * 28  # Размер входного изображения (28x28 пикселей)
output_size = 10  # Количество классов (цифры от 0 до 9)

# Установка параметров обучения
print("Обучение: однослойный персептрон с использованием стохастического градиентного спуска")
print("")

# Загрузка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Определение структуры нейронной сети
class SimplePerceptron(nn.Module):
    def __init__(self):
        super(SimplePerceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, output_size)  # Входной слой: 28*28 пикселей, выходной слой: 10 классов

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

# Инициализация нейронной сети
model = SimplePerceptron()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Параметры для остановки обучения
epsilon_threshold = 0.01  # Порог ошибки для остановки
max_epochs = 10  # Максимальное количество эпох
convergence_epochs = 3  # Количество эпох для проверки сходимости

# Обучение нейронной сети
print("\nНачало обучения:")
print("Эпоха\tОшибка")
for epoch in range(max_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    average_loss = running_loss / len(trainloader)
    print(f"{epoch + 1}\t{average_loss}")

    # Проверка критериев остановки
    if average_loss < epsilon_threshold:
        print(f"Критерий остановки: достигнут порог ошибки {epsilon_threshold}")
        break

    if epoch > convergence_epochs:
        recent_losses = [running_loss / len(trainloader) for _ in range(convergence_epochs)]
        if max(recent_losses) - min(recent_losses) < 0.001:
            print("Критерий остановки: ошибка стабилизировалась в течение нескольких эпох")
            break

# Оценка точности на тестовом наборе
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nТочность на тестовом наборе данных: {100 * correct / total}%")

# Тестирование на случайном изображении из тестового набора
print("\nТестирование на случайном изображении из тестового набора:")
print("Истинная метка\tПредсказанная метка")
random_index = random.randint(0, len(testset) - 1)
image, label = testset[random_index]
output = model(image.unsqueeze(0))  # Добавление размерности батча: (1, 1, 28, 28)
_, predicted = torch.max(output, 1)
print(f"{label}\t\t{predicted.item()}")
