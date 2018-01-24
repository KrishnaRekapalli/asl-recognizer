import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    """
    Other references
    Udacity Forums: https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        BIC_scores = []

        #min_seq = min([len(seq) for seq in self.sequences])
        #self.max_n_components = min (self.max_n_components, min_seq)

        #print("###################",self.max_n_components)

        for n_comps in range(self.min_n_components,self.max_n_components+1):

            try:

                #model = GaussianHMM(n_components=n_comps, covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                model = self.base_model(n_comps)

                logL = model.score(self.X,self.lengths)

                p = n_comps*n_comps + 2*n_comps*len(self.X[0])-1

                bic = (-2)* logL + math.log(len(self.X))*p

            except Exception as e:

                bic = float('inf')


            BIC_scores.append(bic)


        min_index = np.argmin(BIC_scores)


        return self.base_model(range(self.min_n_components,self.max_n_components+1)[min_index])





        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    """
    Other references:
    Udacity Forums: https://discussions.udacity.com/t/dic-score-calculation/238907

    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        DIC_scores = []


        current_word = self.this_word

        rest_of_words = [x for x in list(self.words.keys()) if x != self.this_word]
        #print(rest_of_words)





        for n_comps in range(self.min_n_components,self.max_n_components+1):

            try:

                model = self.base_model(n_comps)
                this_word_score = model.score(self.X,self.lengths)


                # get the average of the model's score on other words
                int_score = 0
                for word in rest_of_words:

                    X, lengths = self.hwords[word]

                    word_score = model.score(X,lengths)

                    int_score = int_score+ word_score

                mean_score_of_other_words = int_score/len(rest_of_words)

                DIC_scores.append(this_word_score-mean_score_of_other_words)

            except Exception as e:
                #print(e)

                DIC_scores.append(float('-inf'))

        max_index = np.argmax(DIC_scores)

        #print(DIC_scores)


        return self.base_model(range(self.min_n_components,self.max_n_components+1)[max_index])


        # TODO implement model selection based on DIC scores



        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    References:
    Udacity Forums: https://discussions.udacity.com/t/implement-selectorcv/247078

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mean_scores = []

        #min_seq = min([len(seq) for seq in self.sequences])
        #self.max_n_components = min (self.max_n_components, min_seq)

        #print("###################",self.max_n_components)

        for n_comps in range(self.min_n_components,self.max_n_components+1):

            try:

                #model = self.base_model(n_comps)




                if len(self.sequences) > 2 :

                    split_method = KFold()
                    int_scores = []

                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                        #self.X, self.lengths = combine_sequences(cv_train_idx,self.sequences)

                        X_train,lengths_train = combine_sequences(cv_train_idx,self.sequences)



                        model = GaussianHMM(n_components=n_comps, covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(X_train, lengths_train)


                        #model = self.base_model(n_comps)

                        X_test,lengths_test = combine_sequences(cv_test_idx,self.sequences)

                        int_scores.append(model.score(X_test,lengths_test))


                    mean_scores.append(np.mean(int_scores))

                else:
                    model = self.base_model(n_comps)

                    mean_scores.append(model.score(self.X,self.lengths))

            except Exception as e :
                #print(e)
                mean_scores.append(float("-inf"))

        #print(mean_scores)
        max_index = np.argmax(mean_scores)

        try:
            return self.base_model(range(self.min_n_components,self.max_n_components+1)[max_index])

        except:

            return None



        # TODO implement model selection using CV
        #raise NotImplementedError
