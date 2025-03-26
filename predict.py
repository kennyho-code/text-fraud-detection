

import torch
from main import LSTMClassifier, create_vocabulary, create_splits, load_dataset
from datasets import load_dataset


def predict_text(model, text, word2idx, device, max_length=50):

    # Set model to evaluation mode
    model.eval()

    # Preprocess the text (same as training)
    text = text.lower()
    words = text.split()

    # Convert words to indices
    indices = [word2idx.get(word, word2idx['<UNK>']) for word in words]

    # Pad or truncate
    if len(indices) < max_length:
        indices += [word2idx['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    # Convert to tensor
    with torch.no_grad():
        input_tensor = torch.tensor(indices).unsqueeze(
            0).to(device)  # Add batch dimension
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)

    return {
        'prediction': 'spam' if prediction.item() == 1 else 'ham',
        'confidence': probabilities[0][prediction].item()
    }


def additional_custom_spam():
    additional_spam = [
        {
            'sms': """Hey this is Jalen with TEKsystems! I just left a voicemail so wanted to know when there is a better time to reach you?""",
            'label': 1  # 1 for spam
        },
        {
            'sms': """(Atlantic Union) Pending $749 transaction on your Checking from #QMB. Wasn't you? Access (aubsave.dnsalias.net) to cancel.""",
            'label': 1
        },
        {
            'sms': """E-ZPass final reminder:
You have an outstanding toll.Your toll account balance is outstanding. If you fail to pay by March 16, 2025. you will face penalties or legal action.
Now Payment:

https://e-zpass.com-emzwsefyx.xin/us

(Please reply Y, then exit the SMS and open it again to activate the link, or copy the link to your Safari browser and open it)
Please settle your toll immediately after reading this message to avoid penalties for delaying the payment. 
Thank you for your cooperation.""",
            'label': 1
        }
        # Add more modern spam examples
    ]


if __name__ == "__main__":
    ds = load_dataset("ucirvine/sms_spam")
    train_dataset, test_dataset = create_splits(ds)
    vocab = create_vocabulary(train_dataset)

    # First recreate the model architecture
    model = LSTMClassifier(len(vocab), 100)
    # Load the saved state dict
    model.load_state_dict(torch.load('best_model.pth'))

    # Move to correct device (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set to evaluation mode
    model.eval()

    test_messages = [
        # "URGENT! You have won a free prize. Call now to claim!",
        # "Hey, what time are we meeting for lunch tomorrow?",
        # "Congratulations! You've been selected for a free iPhone! Click here",
        # "Can you pick up some milk on your way home?",
        "Hey this is Jalen with TEKsystems! I just left a voicemail so wanted to know when there is a better time to reach you?",
        "(Atlantic Union) Pending $749 transaction on your Checking from #QMB. Wasn't you? Access (aubsave.dnsalias.net) to cancel.",
        """E-ZPass final reminder:
You have an outstanding toll.Your toll account balance is outstanding. If you fail to pay by March 16, 2025. you will face penalties or legal action.
Now Payment:

https://e-zpass.com-emzwsefyx.xin/us

(Please reply Y, then exit the SMS and open it again to activate the link, or copy the link to your Safari browser and open it)
Please settle your toll immediately after reading this message to avoid penalties for delaying the payment. 
Thank you for your cooperation.""",
        """E-ZPass Final Reminder:
You have an outstanding toll.Your toll account balance is outstanding. If you fail to pay by March 11, 2025, you will face penalties or legal action.
Now Payment:

https://paytolllmellk.vip/ezdrivema

(Please reply Y, then exit the SMS and open it again to activate the link, or copy the link to your Safari browser and open it)
Please settle your toll immediately after reading this message to avoid penalties for delaying the payment. Thank you for your cooperation"""
    ]

    for message in test_messages:
        result = predict_text(model, message, vocab, device)
        print(f"\nMessage: {message}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
