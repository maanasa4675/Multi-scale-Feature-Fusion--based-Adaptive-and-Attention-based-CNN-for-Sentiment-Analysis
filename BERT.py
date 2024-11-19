import numpy as np
import torch
from tinycss2 import tokenizer
# from embedding4bert.test_xlnet_embedding import tokenizer
from transformers import BertTokenizer, BertModel
#
# # Load the pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Define a sample input sentence
# input_sentence = np.load('Pre_Data.npy', allow_pickle= True)
#
# # Tokenize the input sentence
# tokenized_input = tokenizer.encode(input_sentence, add_special_tokens=True)
#
# # Convert the tokenized input to a PyTorch tensor
# input_tensor = torch.tensor([tokenized_input])
#
# # Pass the input tensor through the BERT model
# with torch.no_grad():
#     outputs = model(input_tensor)
#
# # Extract the output features from the BERT model
# last_hidden_states = outputs[0]
#
# # Print the output features
# print(last_hidden_states)






from transformers import BertTokenizer, BertModel
import torch
an = 1
if an == 1:
    BERT = []
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    text = np.load('Pre_Data.npy', allow_pickle=True)
    for i in range(100):
        # file = text[0][0]
        # for j in range(100):
        tokens = tokenizer.tokenize([i])
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Convert tokens to token IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension

        # Perform forward pass through the BERT model
        outputs = model(token_ids_tensor)

        # Extract the last hidden state of the BERT model
        last_hidden_state = outputs.last_hidden_state

        BERT.append(last_hidden_state)
    np.save('BERT.npy', BERT)




# import torch
# from transformers import BertTokenizer, BertModel
#
# # Load pre-trained BERT model and tokenizer
# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
#
# # Input text
# text = np.load('Pre_Data.npy', allow_pickle=True)
#
# # Tokenize input text
# tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
#
# # Forward pass through BERT model
# outputs = model(**tokens)
#
# # Extract the hidden states and attention weights from the BERT model output
# hidden_states = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)
# attention_weights = outputs.attentions      # list of shape: (batch_size, num_heads, sequence_length, sequence_length)
#
# # Print the shape of hidden states
# print("Hidden states shape:", hidden_states.shape)



#
# import torch
# from transformers import BertTokenizer, BertModel
#
# # Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Input sentence
# sentence =np.load('Pre_Data.npy', allow_pickle= True)
#
# # Tokenize input sentence
# tokens = tokenizer.tokenize(sentence)
#
# # Convert tokens to input IDs
# input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
# # Convert input IDs to tensor
# input_tensor = torch.tensor([input_ids])
#
# # Obtain BERT embeddings
# with torch.no_grad():
#     outputs = model(input_tensor)
#
# # Extract the embeddings
# embeddings = outputs[0]
#
# # Print the shape of the embeddings tensor
# print(embeddings.shape)










# import torch
# from transformers import BertTokenizer, BertModel
#
# # Load pre-trained model and tokenizer
# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
#
# # Input text
# input_text = np.load('Pre_Data.npy', allow_pickle= True)
#
# # Tokenize input text
# tokens = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors='pt')
#
# # Forward pass through BERT model
# outputs = model(**tokens)
#
# # Extract the last hidden state of the tokenizer output
# last_hidden_state = outputs.last_hidden_state
#
# # Convert tensors to numpy arrays
# last_hidden_state = last_hidden_state.detach().numpy()
#
# # Print the shape of the last hidden state
# print("Shape of last hidden state:", last_hidden_state.shape)




#
# import torch
# from transformers import BertModel, BertTokenizer
#
# # Load pre-trained BERT model and tokenizer
# model_name = 'bert-base-uncased'  # or 'bert-base-cased' for a cased model
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
# file = np.load('Pre_Data.npy', allow_pickle= True)
#
# # Example input text
# text = file[0]
#
# # Tokenize the input text
# tokens = tokenizer.tokenize(text)
# input_ids = tokenizer.convert_tokens_to_ids(tokens)
# input_tensor = torch.tensor([input_ids])
#
# # Obtain the BERT model's output
# with torch.no_grad():
#     model.eval()
#     outputs = model(input_tensor)
#
# # Get the output features
# last_hidden_state = outputs.last_hidden_state





#
#
# from transformers import BertTokenizer, BertModel
#
# # Load the pre-trained BERT model and tokenizer
# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name, output_hidden_states=True)
#
# # Example sentence
# sentence = np.load('Pre_Data.npy', allow_pickle= True)
#
# # Tokenize the sentence
# tokens = tokenizer.tokenize(sentence[0])
# input_ids = tokenizer.convert_tokens_to_ids(tokens)
# input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
#
# # Convert the input to PyTorch tensors
# import torch
# input_ids = torch.tensor([input_ids])
#
# # Forward pass through the model
# outputs = model(input_ids)
#
# # Get the hidden states from the model
# hidden_states = outputs.hidden_states
#
# # Extract the features from the second-to-last layer (the second-to-last hidden state)
# features = hidden_states[-2][0]
#
# # Print the extracted features
# print(features)
#
