import matplotlib.pyplot as plt


def generate_feature_importance_plot(importance: dict, filename="feature_importance.png", top_n: int = 20) -> str:
    if not importance:
        raise ValueError("No importance scores provided")

    sorted_items = sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    features, scores = zip(*sorted_items)

    plt.figure(figsize=(8, max(4, len(features) * 0.3)))
    plt.barh(range(len(features)), scores, color="teal")
    plt.yticks(range(len(features)), features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()

    return filename


def generate_residual_plot(y_true, y_pred, filename="residual_plot.png") -> str:
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, y_true - y_pred, alpha=0.3, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close() 
    
    return filename