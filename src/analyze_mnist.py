import numpy as np
import struct
import matplotlib.pyplot as plt
from collections import Counter

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

print("=" * 60)
print("MNIST DATASET ANALYSIS")
print("=" * 60)

train_images = read_idx_images('data/train-images.idx3-ubyte')
train_labels = read_idx_labels('data/train-labels.idx1-ubyte')
test_images = read_idx_images('data/t10k-images.idx3-ubyte')
test_labels = read_idx_labels('data/t10k-labels.idx1-ubyte')

print("\n1. DATASET STRUCTURE")
print("-" * 60)
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Image dimensions: {train_images.shape[1]}x{train_images.shape[2]} pixels")
print(f"Pixel value range: [{train_images.min()}, {train_images.max()}]")

print("\n2. CLASS DISTRIBUTION")
print("-" * 60)
train_counter = Counter(train_labels)
test_counter = Counter(test_labels)

print("\nTraining set:")
for digit in sorted(train_counter.keys()):
    count = train_counter[digit]
    percentage = (count / len(train_labels)) * 100
    print(f"  Digit {digit}: {count:5d} samples ({percentage:.2f}%)")

print("\nTest set:")
for digit in sorted(test_counter.keys()):
    count = test_counter[digit]
    percentage = (count / len(test_labels)) * 100
    print(f"  Digit {digit}: {count:5d} samples ({percentage:.2f}%)")

print("\n3. PIXEL STATISTICS")
print("-" * 60)
print(f"Training set mean pixel value: {train_images.mean():.2f}")
print(f"Training set std pixel value: {train_images.std():.2f}")
print(f"Test set mean pixel value: {test_images.mean():.2f}")
print(f"Test set std pixel value: {test_images.std():.2f}")

non_zero_pixels_train = (train_images > 0).sum(axis=(1, 2)).mean()
non_zero_pixels_test = (test_images > 0).sum(axis=(1, 2)).mean()
print(f"Avg non-zero pixels per image (train): {non_zero_pixels_train:.2f}")
print(f"Avg non-zero pixels per image (test): {non_zero_pixels_test:.2f}")

print("\n4. SAMPLE VISUALIZATION")
print("-" * 60)
print("Generating sample images...")

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
fig.suptitle('MNIST Sample Images (20 random training samples)', fontsize=14)

for i in range(20):
    ax = axes[i // 10, i % 10]
    idx = np.random.randint(0, len(train_images))
    ax.imshow(train_images[idx], cmap='gray')
    ax.set_title(f'Label: {train_labels[idx]}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
print("Saved: mnist_samples.png")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Average Digit Images per Class', fontsize=14)

for digit in range(10):
    ax = axes[digit // 5, digit % 5]
    digit_images = train_images[train_labels == digit]
    avg_image = digit_images.mean(axis=0)
    ax.imshow(avg_image, cmap='gray')
    ax.set_title(f'Digit {digit}', fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_average_digits.png', dpi=150, bbox_inches='tight')
print("Saved: mnist_average_digits.png")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
train_counts = [train_counter[i] for i in range(10)]
test_counts = [test_counter[i] for i in range(10)]

x = np.arange(10)
width = 0.35

ax.bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
ax.bar(x + width/2, test_counts, width, label='Test', alpha=0.8)

ax.set_xlabel('Digit Class', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('MNIST Class Distribution', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(range(10))
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: mnist_distribution.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
