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
   """
   References:
   udacity forums: https://discussions.udacity.com/t/recognizer-implementation/234793/5
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []


    for item in test_set.get_all_sequences():

        X,length = test_set.get_item_Xlengths(item)

        word_scores = {}


        for word, hmm_model in models.items():

            try:

                score = hmm_model.score(X,length)
                word_scores[word] = score

            except:
                word_scores[word] = float('-inf')


        probabilities.append(word_scores)
        guesses.append(max(word_scores, key=word_scores.get))


    return probabilities,guesses




    # TODO implement the recognizer
    # return probabilities, guesses
    raise NotImplementedError
