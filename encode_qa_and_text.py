import argparse
import os
import re
import sys

import numpy as np
import torch
import word2vec as w2v

sys.path.append('.')
sys.path.append('./dataloader')
from dataloader import data_loader as MovieQA
from model import PositionEncoder as pe

mqa = MovieQA.DataLoader()

QA_DESC_TEMPLATE = 'data/descriptor_cache/qa.%s/%s.npy'  # descriptor, qid

re_alphanumeric_upper = re.compile('[^a-zA-Z0-9 -]+')
re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')


def check_save_directory(filename=None, dirname=None):
    """Make the folder where descriptors are saved if it doesn't exist.
    """

    if filename:
        dirname = filename.rsplit('/', 1)[0]

    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def normalize_alphanumeric_with_uppercase(line):
    """Strip all punctuation, keep only alphanumerics.
    """

    line = re_alphanumeric_upper.sub('', line)
    line = re_multispace.sub(' ', line)
    return line


def normalize_alphanumeric(line):
    """Strip all punctuation, keep only alphanumerics.
    """

    line = re_alphanumeric.sub('', line)
    line = re_multispace.sub(' ', line)
    return line


def encode_casings(word):
    pronouns = ["i", "you", "he", "she", "it", "we", "they",
                'me', 'him', 'her', 'us', 'them',
                'yours', 'mine', 'theirs', 'ours', 'hers', 'his', 'its',
                'my', 'your', 'his', 'her', 'its', 'our', 'their']

    casing = np.zeros(5, dtype=int)

    if any(letter.isupper() for letter in word):
        casing[0] = 1
    if any(letter.isnumeric() for letter in word):
        casing[1] = 1
    if word.lower() in pronouns:
        casing[2] = 1

    return casing


def get_positional_encoding(dim, sentence_length):
    div_term = np.power(10000.0, - (np.arange(dim) // 2).astype(np.float32) * 2.0 / dim)
    div_term = div_term.reshape(1, -1)
    pos = np.arange(sentence_length, dtype=np.float32).reshape(-1, 1)
    encoded_vec = np.matmul(pos, div_term)
    encoded_vec[:, 0::2] = np.sin(encoded_vec[:, 0::2])
    encoded_vec[:, 1::2] = np.cos(encoded_vec[:, 1::2])

    return encoded_vec.reshape([sentence_length, dim])


def encode_casing_features(question_list, answer_list, max_sent_len):
    desc_dim = 5
    sentence_list = []
    sentence_list.extend(question_list)
    sentence_list.extend(answer_list)

    features = np.zeros((len(sentence_list), max_sent_len, desc_dim), dtype='int')

    # check for upper, numeric and pronouns
    for s, sentence in enumerate(sentence_list):
        # NOTE: use only alphanumeric normalization, no stemming
        sentence = normalize_alphanumeric_with_uppercase(sentence).split(' ')
        # for each word in the normalized sentence
        for w, word in enumerate(sentence):
            try:
                features[s, w] = encode_casings(word)
            except ValueError:
                print(ValueError)
    # todo : check for unigram/bigram features

    return features[0], features[1:]


def encode_qa_sentences(desc, question_list, answer_list, model, max_sent_len, imdb_key=None, is_qa=False):
    """Encode a list of sentences given the model.
    """
    features = np.zeros((1, 1, 1), dtype='float32')
    sentence_list = []
    sentence_list.extend(question_list)
    sentence_list.extend(answer_list)

    if desc == 'word2vec':
        desc_dim = model.get_vector(model.vocab[-1]).shape[0]
        features = np.zeros((len(sentence_list), max_sent_len, desc_dim), dtype='float32')
        for s, sentence in enumerate(sentence_list):
            # NOTE: use only alphanumeric normalization, no stemming
            sentence = normalize_alphanumeric(sentence.lower()).split(' ')
            # for each word in the normalized sentence
            for w, word in enumerate(sentence):
                if word not in model.vocab: continue
                features[s, w] = model.get_vector(word)

            features[s] /= (np.linalg.norm(features[s]) + 1e-6)

    elif desc == 'GloVe':

        desc_dim = len(model[0][0])
        features = np.zeros((len(sentence_list), max_sent_len, desc_dim), dtype='float32')

        #  Apply GloVe

        for s, sentence in enumerate(sentence_list):
            # NOTE: use only alphanumeric normalization, no stemming
            sentence = normalize_alphanumeric(sentence.lower()).split(' ')
            # for each word in the normalized sentence
            for w, word in enumerate(sentence):

                if word in model[1]:
                    features[s, w] = model[0][model[1][word]]

            features[s] /= (np.linalg.norm(features[s]) + 1e-6)

    return features[0], features[1:]


def encode_qa(desc, model):
    """Encode question and answer using the descriptor.
    """
    QA = mqa.get_qa_data('full')
    check_save_directory(filename=QA_DESC_TEMPLATE % (desc, ''))
    max_len = 32

    ### Count maximum number of word in a sentence
    # for i,qa in enumerate(QA):
    #     sentence = normalize_alphanumeric(qa.question.lower()).split(' ')
    #     if max_len < len(sentence):
    #         max_len = len(sentence)
    #     for ans in qa.answers:
    #         sentence = normalize_alphanumeric(ans.lower()).split(' ')
    #         if max_len < len(sentence):
    #             max_len = len(sentence)

    for i, qa in enumerate(QA):
        npy_fname = QA_DESC_TEMPLATE % (desc, qa.qid)

        # if previously computed then  continue
        if os.path.exists(npy_fname): continue

        # create a list of sentences
        question_list = [qa.question]
        answer_list = [ans for ans in qa.answers if ans]

        # encode sentences, and save features
        question_features, answer_features = encode_qa_sentences(desc, question_list, answer_list, model, max_len,
                                                                 imdb_key=qa.imdb_key, is_qa=True)
        question_casings, answer_casings = encode_casing_features(question_list, answer_list, max_len)

        np.save(npy_fname, [question_features, answer_features, question_casings, answer_casings])


def one_pass_encoding(model, desc):
    """Encode all questions and story types using this model.
    """
    ### Encode all questions
    print('Encoding QA | desc: %s' % (desc))
    encode_qa(desc, model)

    # todo:  Encode all documents
    # for doc in reversed(documents):
    #     print('Encoding %s | desc: %s' %(doc.upper(), desc))
    #     encode_documents(doc, desc, model)


def load_glove_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='data/glove_data/vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='data/glove_data/vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = list(map(float, vals[1:]))

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    return W_norm, vocab, ivocab


def load_glove_embeddings(path, word2idx, embedding_dim):
    """Loading the glove embeddings"""
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                if vector.shape[-1] != embedding_dim:
                    raise Exception('Dimension not matching.')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()

def load_model(desc):
    model = None
    if desc == 'GloVe':
        W_norm, vocab, ivocab = load_glove_params()
        model = [W_norm, vocab, ivocab]

    elif desc == 'word2vec':
        model = w2v.load('model/movie_plots_1364.d-300.mc1.w2v', kind='bin')

    return model, desc


if __name__ == '__main__':
    # desc ='word2vec'
    desc = 'GloVe'
    documents = ['split_plot', 'script', 'subtitle', 'dvs']

    model, desc = load_model(desc)
    one_pass_encoding(model, desc)

    vocab_size = len(model[1])
    emb_dim = model[0].shape[0]
    max_len = 32
    batch_size = 1
    emb_mat = torch.from_numpy(model[0]).float()

    pos_enc = pe.PositionalEncoder(vocab_size, emb_dim, max_len, batch_size, emb_mat)

    QA = mqa.get_qa_data('full')
    sentence = normalize_alphanumeric(QA[100].question.lower()).split(' ')
    sentence_numeric = []
    for i in sentence:
        sentence_numeric.append(model[1][i.lower()])

    print(pos_enc(torch.tensor(sentence_numeric)))
