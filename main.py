from data_preprocessing import load_and_filter_data
from embeddings import generate_embeddings
from model import SimpleNN, train_model, evaluate_model
import torch

def main():
    df, X_train, X_test, y_train, y_test = load_and_filter_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_embeddings = generate_embeddings(X_train)
    X_test_embeddings = generate_embeddings(X_test)

    model = SimpleNN(input_dim=768, output_dim=y_train.shape[1]).to(device)

    train_model(model, X_train_embeddings, y_train)

    f1 = evaluate_model(model, X_test_embeddings, y_test)
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
