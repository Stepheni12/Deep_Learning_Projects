import torch
from transformers import AutoModel

class ImageEncoderCNN(torch.nn.Module):
    """The model extracts the correct sized feature vector from input images."""

    def __init__(self, embed_size):
        """Load pretrained hugging face model and add linear layer."""

        super(ImageEncoderCNN, self).__init__()

        self.model = AutoModel.from_pretrained("apple/mobilevit-xx-small")
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 320 is the size the pooler_output that I'm using as the last layer of the pre-trained.
        # Passed through this liner layer to get to the same size as the word embeddings so I can pass it to the LSTM.
        # Future: Maybe look into using different layer from pre-trained model as the output
        self.linear = torch.nn.Linear(320, embed_size)
        
    def forward(self, images):
        outputs = self.model(images)
        embeddings = outputs.pooler_output
        image_embeddings = self.linear(embeddings)
        
        return image_embeddings # Shape: (batch_size, embed_size)

class CaptionDecoderRNN(torch.nn.Module):
    """The model decodes the image feature vectors and produces captions."""

    def __init__(self, word_embeddings, hidden_size, embedding_size, vocab_size, max_seq_length=25):
        """Load pretrained embeddings, add LSTM layer, and add linear layer output representing the entire vocab."""

        super(CaptionDecoderRNN, self).__init__()

        self.embed = torch.nn.Embedding.from_pretrained(word_embeddings, freeze=True) # Future: Try unfreezing?
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        
    def forward(self, image_embeddings, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((image_embeddings.unsqueeze(1), embeddings), 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        return outputs # Shape: (batch_size, vocab_size)
    

    def sample(self, features, states=None):
        """Generate words to produce captions for given image features using greedy search."""

        # Previous predicted output is used as the next input
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids