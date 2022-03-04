import random
import json
import torch
import re
import numpy as np
from model import NeuralNet, AdvancedNeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

# Get Back the Information we need for our model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
bert = data["bert"]
tokenizer = data["tokenizer"]
max_seq_len = data["max_seq_len"]
le = data["le"]


model = AdvancedNeuralNet(
    bert, input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)


def get_prediction(str):
    str = re.sub(r"[ ^ a-zA-Z] +", "", str)
    test_text = [str]
    model.eval()

    tokens_test_data = tokenizer(
        test_text,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data["input_ids"])
    test_mask = torch.tensor(tokens_test_data["attention_mask"])

    preds = None
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    print("Intent Identified: ", le.inverse_transform(preds)[0])
    return le.inverse_transform(preds)[0]


def get_response(message):
    intent = get_prediction(message)
    for i in data['intents']:
        if i["tag"] == intent:
            result = random.choice(i["responses"])
            break
    print(f"Response : {result}")
    return "Intent: " + intent + '\n' + "Response: " + result


# Creating the chat
bot_name = "Dinkleberg"
print("Let's chat! type 'quit' to leave")
# Note, this is all information we pulled from above keys
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    tag = get_prediction(sentence)
    for intent in intents['intents']:
        if tag == intent['tag']:
            # Randomly select a response from the intent's response list
            response = random.choice(intent['responses'])
            print(bot_name + ': ' + response)

    sentence = tokenize(sentence)
    # X = bag_of_words(sentence, all_words)
    # X = X.reshape(1, X.shape[0])
    # X = torch.from_numpy(X)
    # output = model(X)
    # _, predicted = torch.max(output, dim=1)
    # tag = tags[predicted.item()]
    # # Want to find the intents, so we loop over through all our intents

    # # Probability of prediction for each tag
    # probs = torch.softmax(output, dim=1)
    # prob = probs[0][predicted.item()]

    # if prob.item() > 0.75:
    #     for intent in intents["intents"]:
    #         if tag == intent["tag"]:
    #             print(f"{bot_name}: {random.choice(intent['responses'])}")
    # else:
    #     print(f"{bot_name}: I do not understand...")
