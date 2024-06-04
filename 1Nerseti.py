import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size, output_size) * 0.006 - 0.003
        self.threshold = 0.0
    
    def activation_function(self, x):
        return np.where(x >= self.threshold, 1, 0)
    
    def forward_pass(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation_function(weighted_sum)
    
    def train(self, inputs, target):
        output = self.forward_pass(inputs)
        error = target - output
        self.weights += self.learning_rate * np.outer(inputs, error)
        return error
    
    def test(self, inputs):
        return self.forward_pass(inputs)


def generate_image(symbol, font):
    image = Image.new('L', (28, 28), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), symbol, font=font, fill=0)
    return np.array(image) / 255.0


def generate_random_image(symbol, font):
    image = Image.new('L', (28, 28), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((5, 5), symbol, font=font, fill=0)
    for _ in range(100):
        x = random.randint(0, 0)
        y = random.randint(0, 0)
        draw.point((x, y), fill=0)
    return np.array(image) / 255.0


font_paths = {
    'A': [r"D:\PYTHON\SETI\fonts\font_A1.ttf", r"D:\PYTHON\SETI\fonts\font_A2.ttf", r"D:\PYTHON\SETI\fonts\font_A3.ttf", r"D:\PYTHON\SETI\fonts\font_A4.ttf"],
    'B': [r"D:\PYTHON\SETI\fonts\font_B1.ttf", r"D:\PYTHON\SETI\fonts\font_B2.ttf", r"D:\PYTHON\SETI\fonts\font_B3.ttf", r"D:\PYTHON\SETI\fonts\font_B4.ttf"],
    'C': [r"D:\PYTHON\SETI\fonts\font_C1.ttf", r"D:\PYTHON\SETI\fonts\font_C2.ttf", r"D:\PYTHON\SETI\fonts\font_C3.ttf", r"D:\PYTHON\SETI\fonts\font_C4.ttf"],
    'D': [r"D:\PYTHON\SETI\fonts\font_D1.ttf", r"D:\PYTHON\SETI\fonts\font_D2.ttf", r"D:\PYTHON\SETI\fonts\font_D3.ttf", r"D:\PYTHON\SETI\fonts\font_D4.ttf"]
}

fonts = {symbol: [ImageFont.truetype(path, 20) for path in paths] for symbol, paths in font_paths.items()}

symbols = ['A', 'B', 'C', 'D']
num_samples_per_symbol = 4

training_data = []
test_data = {}
random_test_data = {}

for symbol in symbols:
    symbol_training_data = []
    for font in fonts[symbol]:
        image_data = generate_image(symbol, font)
        flattened_data = image_data.flatten()
        training_data.append(np.append(flattened_data, symbols.index(symbol)))
        symbol_training_data.append(flattened_data)
    test_data[symbol] = generate_image(symbol, random.choice(fonts[symbol]))
    random_test_data[symbol] = generate_random_image(symbol, random.choice(fonts[symbol]))

np.random.shuffle(training_data)

input_size = 28 * 28
output_size = len(symbols)
learning_rate = 0.1
perceptron = Perceptron(input_size, output_size, learning_rate)

num_epochs = 100
for epoch in range(num_epochs):
    total_error = 0
    for data in training_data:
        inputs = data[:-1]
        target = data[-1]
        error = perceptron.train(inputs, target)
        total_error += np.abs(error)
    accuracy = 1 - (total_error / (len(training_data) * perceptron.output_size))
    if epoch % 2 == 0:
        print(f"Эпоха {epoch+1}/{num_epochs}, Точность: {accuracy[0]:.2f}\tЭпоха {epoch+2}/{num_epochs}, Точность: {accuracy[1]:.2f}")
    print(f"Эпоха {epoch+1}/{num_epochs}, Суммарная ошибка: {total_error.sum():.2f}")

correct_predictions = 0
for symbol, data in test_data.items():
    inputs = data.flatten()
    prediction = perceptron.test(inputs)
    predicted_class = np.argmax(prediction)
    actual_class = symbols.index(symbol)
    if predicted_class == actual_class:
        correct_predictions += 1
        print(f"Предсказано: {symbols[predicted_class]}, Фактическое: {symbol}")
    else:
        print(f"Предсказано: {symbols[predicted_class]}, Фактическое: {symbol}")
accuracy = correct_predictions / len(test_data)
print(f"Точность на тесте: {accuracy}")

for symbol, data in random_test_data.items():
    inputs = data.flatten()
    prediction = perceptron.test(inputs)
    predicted_class = np.argmax(prediction)
    plt.imshow(data.reshape(28, 28), cmap='gray')
    plt.title(f"Случайный тест - Предсказано: {symbols[predicted_class]}, Фактическое: {symbol}")
    plt.show()
