import numpy as np
# from gensim.models import Word2Vec
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')



def train_word2vec(data: str) -> dict:
    vector_size = 100
    window_size = 5
    epochs = 10
    learning_rate = 0.01

    words = data.strip().split()
    vocabulary_size = len(words)
    vocab = list(set(words))

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}

    W1 = np.random.uniform(-0.1, 0.1, (vocabulary_size, vector_size)) #weihts for input
    W2 = np.random.uniform(-0.1, 0.1, (vector_size, vocabulary_size))   # weights for output

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def one_hot(idx, size):
        v = np.zeros(size)
        v[idx] = 1.0
        return v

    pairs = []
    for i, word in enumerate(words):
        center = word2idx[word]
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                contex = word2idx[words[j]]
                pairs.append((center, contex))

    for _ in range(epochs):
        for center, contex in pairs:
            h = W1[center] # embeddings
            u = np.dot(W2.T, h)
            y_pred = softmax(u)
            y_true = one_hot(contex, vocabulary_size)
            error = y_pred - y_true

            #BACKPROP
            dw2 = np.outer(h, error) #outer multyplication
            dw1 = np.dot(W2, error)

            #upd weihts

            W2 -= learning_rate * dw2
            W1[center] -= learning_rate * dw1

    w2v_dict = {word: W1[word2idx[word]] for word in vocab}

    return w2v_dict


data = "your string hello world your hello"
w2v_dict = train_word2vec(data)
print(w2v_dict)




# def train_word2vec(data: str) -> dict:
#     # 1. Токенизация — разбиваем строку на предложения/слова
#     sentences = [sentence.split() for sentence in data.strip().split('\n') if sentence.strip()]
#
#     # Если весь текст — одна строка, разбиваем по словам в одно "предложение"
#     if not sentences:
#         sentences = [data.strip().split()]
#
#     # sentences = word_tokenize(data)
#
#     # 2. Обучаем модель Word2Vec
#     model = Word2Vec(
#         sentences=sentences,
#         vector_size=100,  # размерность эмбеддинга
#         window=5,  # размер контекстного окна
#         min_count=1,  # минимальная частота слова (1 = берём все слова)
#         workers=4,  # количество потоков
#         sg=1,  # 1 = Skip-gram, 0 = CBOW
#         epochs=10  # количество эпох обучения
#     )
#
#     # 3. Формируем словарь {слово: numpy_array}
#     w2v_dict = {word: np.array(model.wv[word]) for word in model.wv.index_to_key}
#
#     return w2v_dict

# data = "your string hello world your hello"
# w2v_dict = train_word2vec(data)
# print(w2v_dict)