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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_features = len(self.X[0])
        n_datapoints = len(self.X)

        best_model = None
        best_BIC = float("inf")

        for n_component in range(self.min_n_components, self.max_n_components+1):
            try:
                loop_model = self.base_model(n_component)
                loop_logL = loop_model.score(self.X, self.lengths) #LogL of current model


                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/8
                # Initial state occupation probabilities = numStates
                # Transition probabilities = numStates*(numStates - 1)
                # Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
                # Also https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt
                loop_params = n_component * (n_component - 1) + (n_component - 1) + \
                             2 * n_features * n_component

                loop_BIC = (-2 * loop_logL) + (loop_params * np.log(n_datapoints)) #use fomula from slide

                if best_BIC > loop_BIC:
                    best_model = loop_model
                    best_BIC = loop_BIC
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n_component))
        return best_model





class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_DIC = float('-inf')

        for n_component in range(self.min_n_components, self.max_n_components):
            loop_model = self.base_model(n_component)
            this_word_logL = 0.0    #log(P(X(i)))
            other_word_logL = 0.0   #SUM(log(P(X(all but i))
            other_word_count = 0.0  #(M-1)

            for word, (X, length) in self.hwords.items():
                try:
                    loop_logL = loop_model.score(X, length)
                    if word == self.this_word:
                        this_word_logL += loop_logL
                    else:
                        other_word_logL += loop_logL
                        other_word_count += 1
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, n_component))
            if other_word_count > 0:
                loop_DIC = this_word_logL - other_word_count/float(other_word_count) #using formula
                if loop_DIC > best_DIC:
                    best_DIC = loop_DIC
                    best_model = loop_model
        return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def get_model_at(self, num_component, index):
        x_train, train_length = combine_sequences(index, self.sequences)
        model = self.base_model(num_component)
        model = model.fit(x_train, train_length)
        return model

    def get_score_at(self, model, index):
        x_test, test_length = combine_sequences(index, self.sequences)
        return model.score(x_test, test_length)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_logL = float("-inf")

        # Split to either 3 states or the length of sequence whichever is smaller
        split_method = KFold( n_splits=min(3,len(self.sequences)) )

        for n_component in range(self.min_n_components, self.max_n_components+1):
            for train_index, test_index in split_method.split(self.sequences):
                try:
                    loop_model = self.get_model_at(n_component, train_index)
                    loop_logL = self.get_score_at(loop_model, test_index)
                    if loop_logL > best_logL:
                        best_model = loop_model
                        best_logL = loop_logL
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, n_component))
        return best_model


# if __name__ == "__main__":
#     from  asl_test_model_selectors import TestSelectors
#     test_model = TestSelectors()
#     test_model.setUp()
    # test_model.test_select_cv_interface()
    # test_model.test_select_bic_interface()
    # test_model.test_select_dic_interface()
