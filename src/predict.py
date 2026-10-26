"""Predict SMS labels and display the fitted model's class probabilities."""
import argparse
from pathlib import Path

import joblib


def predict_sms(model, message: str) -> dict:
    """Classify a nonblank message using the saved pipeline without refitting."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a nonblank SMS message.")
    probabilities = model.predict_proba([message])[0]
    by_label = {int(label): float(value) for label, value in zip(model.classes_, probabilities)}
    predicted = int(model.predict([message])[0])
    recognized = model.named_steps["vectorizer"].transform([message]).nnz > 0
    return {
        "message": message,
        "prediction": "Spam" if predicted == 1 else "Ham",
        "p_spam": by_label[1],
        "p_ham": by_label[0],
        "has_known_words": bool(recognized),
    }


def format_prediction(result: dict) -> str:
    lines = [
        f"Prediction: {result['prediction']}",
        f"P(Spam | Message) = {result['p_spam']:.10g}",
        f"P(Ham | Message) = {result['p_ham']:.10g}",
    ]
    if not result["has_known_words"]:
        lines.append("No recognized vocabulary words; this prediction uses class priors only.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Classify SMS messages with the trained Naive Bayes pipeline.")
    parser.add_argument("--message", help="Classify a single SMS instead of opening the input prompt.")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "models/sms_naive_bayes.joblib", help="Path to a trusted saved model pipeline.")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Saved model not found. Run the project notebook first.")
    model = joblib.load(args.model)
    if args.message is not None:
        try:
            print(format_prediction(predict_sms(model, args.message)))
        except ValueError as exc:
            parser.error(str(exc))
        return
    print("SMS Spam Detection — enter a message, or type /quit to exit.")
    while True:
        try:
            message = input("SMS: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if message.strip().lower() == "/quit":
            break
        try:
            print(format_prediction(predict_sms(model, message)))
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    main()
