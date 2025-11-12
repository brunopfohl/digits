import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

def create_knn_concept_visualization():
    """Visualize KNN concept with simple 2D example"""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    class_0 = np.random.randn(20, 2) + np.array([2, 2])
    class_1 = np.random.randn(20, 2) + np.array([6, 6])
    new_point = np.array([4.5, 5.0])

    for idx, k in enumerate([1, 3, 7]):
        ax = axes[idx]

        ax.scatter(class_0[:, 0], class_0[:, 1], c='blue', s=100,
                   alpha=0.6, label='Class 0', edgecolors='black')
        ax.scatter(class_1[:, 0], class_1[:, 1], c='red', s=100,
                   alpha=0.6, label='Class 1', edgecolors='black')
        ax.scatter(new_point[0], new_point[1], c='green', s=300,
                   marker='*', edgecolors='black', linewidth=2,
                   label='New Point', zorder=5)

        all_points = np.vstack([class_0, class_1])
        all_labels = np.array([0]*20 + [1]*20)

        distances = np.sqrt(np.sum((all_points - new_point)**2, axis=1))
        nearest_indices = np.argsort(distances)[:k]

        for i in nearest_indices:
            ax.plot([new_point[0], all_points[i, 0]],
                   [new_point[1], all_points[i, 1]],
                   'g--', alpha=0.5, linewidth=2)
            circle = Circle(all_points[i], 0.3, color='green',
                          fill=False, linewidth=3)
            ax.add_patch(circle)

        votes_0 = np.sum(all_labels[nearest_indices] == 0)
        votes_1 = np.sum(all_labels[nearest_indices] == 1)
        prediction = 0 if votes_0 > votes_1 else 1
        pred_color = 'blue' if prediction == 0 else 'red'

        ax.set_title(f'K={k}: Class 0: {votes_0} votes, Class 1: {votes_1} votes\n'
                    f'Prediction: Class {prediction}',
                    fontsize=12, fontweight='bold', color=pred_color)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 9)

    plt.suptitle('K-Nearest Neighbors: Effect of K Parameter',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('knn_concept.png', dpi=150, bbox_inches='tight')
    print("Saved: knn_concept.png")
    plt.close()

def create_distance_comparison():
    """Visualize different distance metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    point_a = np.array([2, 2])
    point_b = np.array([8, 7])

    ax = axes[0]
    ax.scatter(*point_a, c='blue', s=200, zorder=5, label='Point A')
    ax.scatter(*point_b, c='red', s=200, zorder=5, label='Point B')
    ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]],
           'g-', linewidth=3, label='Euclidean Distance')
    ax.plot([point_a[0], point_b[0], point_b[0]],
           [point_a[1], point_a[1], point_b[1]],
           'orange', linewidth=3, linestyle='--', label='Manhattan Distance')

    euclidean = np.sqrt((point_b[0]-point_a[0])**2 + (point_b[1]-point_a[1])**2)
    manhattan = abs(point_b[0]-point_a[0]) + abs(point_b[1]-point_a[1])

    ax.text(5, 3, f'Euclidean: {euclidean:.2f}', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.text(5, 5.5, f'Manhattan: {manhattan:.2f}', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Distance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    x = np.linspace(-10, 10, 1000)
    y_l1 = np.maximum(0, 5 - np.abs(x))
    y_l2_pos = np.sqrt(np.maximum(0, 25 - x**2))
    y_l2_neg = -y_l2_pos

    ax = axes[1]
    ax.fill_between(x, -5+np.abs(x), 5-np.abs(x), alpha=0.3,
                    color='orange', label='Manhattan (L1)')
    ax.plot(x[y_l2_pos >= 0], y_l2_pos[y_l2_pos >= 0], 'g-', linewidth=3)
    ax.plot(x[y_l2_pos >= 0], y_l2_neg[y_l2_pos >= 0], 'g-',
           linewidth=3, label='Euclidean (L2)')
    ax.fill_between(x[y_l2_pos >= 0], y_l2_neg[y_l2_pos >= 0],
                    y_l2_pos[y_l2_pos >= 0], alpha=0.3, color='green')

    ax.scatter(0, 0, c='red', s=200, zorder=5, marker='*')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Points at Distance 5 from Origin', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('distance_metrics.png', dpi=150, bbox_inches='tight')
    print("Saved: distance_metrics.png")
    plt.close()

def create_k_value_effect():
    """Visualize overfitting vs underfitting with K"""
    k_values = [1, 3, 5, 7, 9, 11, 15, 20, 30, 50]
    train_acc = [1.0, 0.99, 0.98, 0.97, 0.96, 0.955, 0.94, 0.93, 0.91, 0.88]
    val_acc = [0.92, 0.96, 0.972, 0.970, 0.968, 0.965, 0.96, 0.955, 0.94, 0.92]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(k_values, train_acc, 'bo-', linewidth=3, markersize=10,
           label='Training Accuracy', alpha=0.7)
    ax.plot(k_values, val_acc, 'ro-', linewidth=3, markersize=10,
           label='Validation Accuracy', alpha=0.7)

    best_k_idx = np.argmax(val_acc)
    best_k = k_values[best_k_idx]
    ax.axvline(x=best_k, color='green', linestyle='--', linewidth=2,
              label=f'Optimal K={best_k}')
    ax.scatter(best_k, val_acc[best_k_idx], c='green', s=300,
              marker='*', zorder=5, edgecolors='black', linewidth=2)

    ax.fill_between([k_values[0], 3], [0.85, 0.85], [1.05, 1.05],
                    alpha=0.2, color='red', label='Overfitting Zone')
    ax.fill_between([20, k_values[-1]], [0.85, 0.85], [1.05, 1.05],
                    alpha=0.2, color='blue', label='Underfitting Zone')

    ax.set_xlabel('K (Number of Neighbors)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Effect of K on Model Performance (Bias-Variance Tradeoff)',
                fontsize=16, fontweight='bold')
    ax.set_ylim(0.85, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='lower left')

    ax.text(1.5, 0.87, 'Too flexible\nHigh variance', fontsize=10,
           ha='center', style='italic')
    ax.text(35, 0.87, 'Too simple\nHigh bias', fontsize=10,
           ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('k_value_effect.png', dpi=150, bbox_inches='tight')
    print("Saved: k_value_effect.png")
    plt.close()

def create_knn_algorithm_flowchart():
    """Create visual flowchart of KNN algorithm"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')

    boxes = [
        (0.5, 0.95, 'START\nNew image to classify', 'lightblue'),
        (0.5, 0.85, 'Flatten image:\n28×28 → 784 pixels', 'lightgreen'),
        (0.5, 0.75, 'Normalize:\nPixels ÷ 255', 'lightgreen'),
        (0.5, 0.65, 'Calculate distance to ALL\n60,000 training images', 'lightyellow'),
        (0.5, 0.55, 'Sort distances:\nSmallest to largest', 'lightyellow'),
        (0.5, 0.45, 'Select K=5\nnearest neighbors', 'lightyellow'),
        (0.5, 0.35, 'Count class labels:\nClass 7: 4 votes\nClass 1: 1 vote', 'lightcoral'),
        (0.5, 0.25, 'Majority vote:\nPredicted class = 7', 'lightcoral'),
        (0.5, 0.15, 'END\nReturn prediction', 'lightblue'),
    ]

    for x, y, text, color in boxes:
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color,
                   edgecolor='black', linewidth=2)
        ax.text(x, y, text, fontsize=12, ha='center', va='center',
               bbox=bbox, weight='bold')

        if y > 0.15:
            ax.annotate('', xy=(x, y-0.08), xytext=(x, y-0.02),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('KNN Algorithm Flowchart for MNIST',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('knn_flowchart.png', dpi=150, bbox_inches='tight')
    print("Saved: knn_flowchart.png")
    plt.close()

def create_computational_complexity():
    """Visualize computational complexity"""
    n_samples = np.array([100, 500, 1000, 5000, 10000, 50000, 60000])
    time_knn = n_samples * 0.001
    time_nn = np.log(n_samples) * 5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(n_samples, time_knn, 'ro-', linewidth=3, markersize=10,
           label='KNN (Linear)', alpha=0.7)
    ax.plot(n_samples, time_nn, 'bo-', linewidth=3, markersize=10,
           label='Neural Network (Constant)', alpha=0.7)
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Time Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1]
    memory_knn = n_samples * 784 * 4 / (1024**2)
    memory_nn = np.ones_like(n_samples) * 5

    ax.plot(n_samples, memory_knn, 'ro-', linewidth=3, markersize=10,
           label='KNN (Stores all data)', alpha=0.7)
    ax.plot(n_samples, memory_nn, 'bo-', linewidth=3, markersize=10,
           label='Neural Network (Fixed weights)', alpha=0.7)
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('computational_complexity.png', dpi=150, bbox_inches='tight')
    print("Saved: computational_complexity.png")
    plt.close()

if __name__ == "__main__":
    print("Generating KNN visualizations for video...")
    print("=" * 60)

    create_knn_concept_visualization()
    create_distance_comparison()
    create_k_value_effect()
    create_knn_algorithm_flowchart()
    create_computational_complexity()

    print("=" * 60)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  1. knn_concept.png - Shows how K affects classification")
    print("  2. distance_metrics.png - Compares different distance metrics")
    print("  3. k_value_effect.png - Shows bias-variance tradeoff")
    print("  4. knn_flowchart.png - Algorithm step-by-step")
    print("  5. computational_complexity.png - Performance comparison")
