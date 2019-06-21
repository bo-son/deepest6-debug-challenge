import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class HierarchialAttentionNetwork(nn.Module):
    """
    Yang et al. (2016). Hierarchical Attention Networks.
    See details for paper: https://www.cs.cmu.edu/%7Ediyiy/docs/naacl16.pdf
    """
    def __init__(self, num_classes, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim):
        """
        :param num_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param embed_dim: dimension of word embeddings
        :param word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        :param sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        :param word_gru_num_layers: number of layers in word-level GRU
        :param sent_gru_num_layers: number of layers in sentence-level GRU
        :param word_att_dim: dimension of word-level attention layer
        :param sent_att_dim: dimension of sentence-level attention layer
        """
        super(HierarchialAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim)

        # classifier
        self.fc = nn.Linear(2 * sent_gru_hidden_dim, num_classes)
        self.out = nn.Softmax(dim=-1)


    def forward(self, docs, doc_lengths, sent_lengths):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, max_sent_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        doc_embeds, word_att_weights, sent_att_weights = self.sent_attention(docs, doc_lengths, sent_lengths)  
        scores = self.out(self.fc(doc_embeds))

        return scores, word_att_weights, sent_att_weights


class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers, word_att_dim)

        # Bidirectional sentence-level GRU
        self.gru = nn.GRU(
            2 * word_gru_hidden_dim, sent_gru_hidden_dim, num_layers=sent_gru_num_layers,
            batch_first=True, bidirectional=True)

        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, docs, doc_lengths, sent_lengths):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, padded_doc_length)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """
        # Sort documents by decreasing order in length
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim=0, descending=True)
        docs = docs[doc_perm_idx]
        sent_lengths = sent_lengths[doc_perm_idx]

        # Make a long batch of sentences by removing pad-sentences
        # i.e. `docs` was of size (num_docs, padded_doc_length, padded_sent_length)
        # -> `packed_sents.data` is now of size (num_sents, padded_sent_length)
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_sents.batch_sizes

        # Make a long batch of sentence lengths by removing pad-sentences
        # i.e. `sent_lengths` was of size (num_docs, padded_doc_length)
        # -> `packed_sent_lengths.data` is now of size (num_sents)
        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=doc_lengths.tolist(), batch_first=True)

        # Word attention module
        sents, word_att_weights = self.word_attention(packed_sents.data, packed_sent_lengths.data)

        # Sentence-level GRU over sentence embeddings
        packed_sents, _ = self.gru(PackedSequence(sents, valid_bsz))

        # Restore as documents by repadding
        docs, _ = pad_packed_sequence(packed_sents, batch_first=True)

        # Sentence attention
        att = torch.tanh(self.sent_attention(packed_sents.data))
        att = self.sentence_context_vector(att).squeeze(1)

        att = torch.exp(att)

        # Restore as documents by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)
        sent_att_weights = att / torch.sum(att)

        # Compute document vectors
        docs = docs * sent_att_weights.unsqueeze(2)
        docs = docs.sum(dim=1)

        # Restore as documents by repadding
        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)

        # Restore the original order of documents (undo the first sorting)
        _, doc_unperm_idx = doc_perm_idx.sort(dim=0, descending=False)
        docs = docs[doc_unperm_idx]

        return docs, word_att_weights, sent_att_weights


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        :param embeddings: embeddings to init with
        """
        pass

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        :param freeze: True if to freeze weights
        """
        pass

    def forward(self, sents, sent_lengths):
        """
        :param sents: encoded sentence-level data; LongTensor (num_sents, pad_len, embed_dim)
        :param sent_lengths: sentence lengths; LongTensor (num_sents)
        :return: sentence embeddings, attention weights of words
        """
        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents)

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)
        
         # effective batch size at each timestep
        valid_bsz = packed_words.batch_sizes

        # Apply word-level GRU over word embeddings
        packed_words, _ = self.gru(packed_words)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        # Word Attenton
        att = torch.tanh(self.attention(packed_words.data))
        att = self.context_vector(att).squeeze(1)

        att = torch.exp(att)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)
        att_weights = att / torch.sum(att)

        # Compute sentence vectors
        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False) 
        sents = sents[sent_unperm_idx] 

        return sents, att_weights