from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
# Bottom ssl is workaround for broken script on punkt donwloadm which returns a loading ssl error
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
# End of error workaround

stemmer = PorterStemmer()
# Imports needed from nltk

# Our Tokenizer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stemming Function


def stem(word):
    return stemmer.stem(word.lower())

# Bag of Words Function


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# TODO: Test our function with the below sentence to visualize Tokenization.
# TODO CONT: What is the purpose of tokenizing our text?
# Tokenizing text is usueful for punctation stripping, creating a data structure for our text, and for word stemming. By breaking sentences down into words, we can create a bag of words representation of our text.
# Testing our Tokenizer
test_sentence = "I will not live in peace until I find the Avatar!"
print("-------------- Tokenizer --------------")
print(tokenize(test_sentence))
print("-------------- End Tokenizer --------------\n")

# TODO: Test our Stemming function on the below words.
# TODO CONT: How does stemming affect our data?
# Stemming breaks down our words into their roots. This makes it easier to compare words and reduces the number of words we need to compare.
words = ["Organize", "organizes", "organizing", "disorganized"]
print("-------------- Stemming --------------")
for w in words:
    print(w, " : ", stem(w))
print("-------------- End Stemming --------------\n")


# TODO: Implement the above Bag of Words function on the below sentence and words.
# TODO (CONTINUED): What does the Bag of Words model do? Why would we use Bag of Words here instead of TF-IDF or Word2Vec?
# the bag of words model represents our words as a vector of 0s and 1s. We use bag of words instead of TF-IDF because we don't need to know the frequency of each word, we just need to see if the word is in the sentence or not.
# We don't use word2vec because we aren't looking for semantic similarity between the words here. However, if we care about the context of the words here, word2vec is a better option.
print("Testing our bag_of_words function")
sentence = ["I", "will", "now", "live", "in",
            "peace", "until", "I", "find", "the", "Avatar"]
words = ["hi", "hello", "I", "you", "the", "bye", "in", "cool", "wild", "find"]
print(bag_of_words(sentence, words))
print("--------------")
