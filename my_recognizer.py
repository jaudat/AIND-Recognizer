import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # https://discussions.udacity.com/t/recognizer-implementation/234793/5

    test_sequences = list(test_set.get_all_Xlengths().values())
    for test_X, test_Xlength in  test_sequences:
        probability = {}
        best_word = None
        best_logL = float("-inf")
        for word, model in models.items():
            try:
                loop_logL = model.score(test_X, test_Xlength)
                probability[word] = loop_logL
                if loop_logL > best_logL:
                    best_word = word
                    best_logL = loop_logL
            except:
                probability[word] = float('-inf')

        guesses.append(best_word)
        probabilities.append(probability)

    # return probabilities, guesses
    return probabilities, guesses

# if __name__ == "__main__":
#     from asl_test_recognizer import TestRecognize
#     test_model = TestRecognize()
#     test_model.setUp()
#     test_model.test_recognize_guesses_interface()
#     test_model.test_recognize_guesses_interface()
