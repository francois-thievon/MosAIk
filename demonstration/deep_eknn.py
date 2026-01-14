from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from lib.utils import noise_imprecision
from lib import ibelief
import faiss

import numpy as np
import math

# Value of the Alpha parameter
ALPHA = 0.4
BETA = 0.5

class EKNN(BaseEstimator, ClassifierMixin):
    """
    EK-NN class used to quantify uncertainty for deep learning applications.
    
    Based on the Evidental k nearest neighbours (EKNN) classifier by Denoeux (1995).
    """

    def __init__(self, class_number, n_neighbors=5):
        """
        EK-NN class used to quantify uncertainty for deep learning applications.

        Parameters
        -----
        n_neighbors : int
            number of nearest neighbors, default = 5

        Returns
        -----
        The instance of the class.
        """

        # Used to retrieve the n nearest neighbors
        self.n_neighbors = n_neighbors

        # Select number of classes
        self.nb_classes = 2**class_number - 1 

        # Used to retrieve the state of the model
        self._fitted = False

    def get_params(self):
        # Return the number of nearest neighbors as a dict
        return {"n_neighbours": self.n_neighbors}

    def set_params(self, n_neighbors):
        # Set the number of nearest neighbors
        self.n_neighbors = n_neighbors

    def score(self, X, y_true, criterion=3):
        """
        Calculate the accuracy score of the model,
        unsig a specific criterion in "Max Credibility", 
        "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions.
        criterion : int
            Choosen criterion for prediction, by default criterion = 3.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        y_pred = self.predict(X, criterion=criterion)

        # Compare with true labels, and compute accuracy
        return accuracy_score(y_true, y_pred)
    
    def fit(self, X, y, alpha=ALPHA, beta=BETA):
        """
        Fit the model according to the training data.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y : ndarray
            Labels array
        alpha : int
            Value of the alpha parameter, default = 0.95
        beta : int
            Value of the beta parameter, default = 1.5
        unique_gamma : boolean
            True for a unique computation of a global gamma parameter, 
            False for multiple gammas (high computational cost). default = True.
        Returns
        -----
        self : EKNN
            The instance of the class.
        """

        # Label encoder 
        self.label_encoder = preprocessing.LabelEncoder().fit(y)

        # Format labels to belief functions
        y = self.label_encoder.transform(y)
        classes = np.array([i for i in range(int(math.log2(self.nb_classes + 1)))])
        _, y = noise_imprecision(y, classes.shape[0], classes, noise=0)
        
        # Check for data integrity
        if X.shape[0] != y.shape[0]:
            if X.shape[0] * (self.nb_classes + 1) == y.shape[0]:
                y = np.reshape(y, (-1, self.nb_classes + 1))
            else:
                raise ValueError("X and y must have the same number of rows")

        # Verify if the size of y is of a power set (and if it contains the empty set or not)
        if math.log(y.shape[1], 2).is_integer():
            y = y[:,1:]
        elif not math.log(y.shape[1] + 1, 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")

        # Save X and y
        self.X_trained = X
        self.y_trained = y

        # Save size of the dataset
        self.size = self.X_trained.shape[0]

        # Init gamma and alpha
        self._init_parameters(alpha=alpha, beta=beta)

        # The model is now fitted
        self._fitted = True

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled

        Returns
        -----
        predictions : ndarray
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        result = self._predict(X)

        predictions = ibelief.decisionDST(result.T, 4, return_prob=True)

        return predictions

    def predict(self, X, criterion=3, return_bba=False):
        """
        Predict labels of input data. Can return all bbas. Criterion are :
        "Max Credibility", "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled
        creterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".
        return_bba : boolean
            Type of return, predictions or both predictions and bbas, 
            by default return_bba=False.

        Returns
        -----
        predictions : ndarray
        result : ndarray
            Predictions if return_bba is False and both predictions and masses if return_bba is True
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        # Predict output bbas for X
        result = self._predict(X)

        # Max Plausibility
        if criterion == 1:
            predictions = ibelief.decisionDST(result.T, 1)
        # Max Credibility
        elif criterion == 2:
            predictions = ibelief.decisionDST(result.T, 2)
        # Max Pignistic probability
        elif criterion == 3:
            predictions = ibelief.decisionDST(result.T, 4)
        else:
            raise ValueError("Unknown decision criterion")

        predictions = self.label_encoder.inverse_transform(predictions)

        # Return predictions or both predictions and bbas
        if return_bba:
            return predictions, result
        else:
            return predictions
        
    def get_uncertainties(self, X):
        """
        Compute aleatoric and epistemic uncertainties of input data.

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled

        Returns
        -----
        aleatoric : ndarray
            Aleatroic uncertainties of the model for each test data.
        epistemic : ndarray
            Epistemic uncertainties of the model for each test data.
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        # Predict output bbas for X
        result = self._predict(X)

        # Compute epistemic
        epistemic = result[:,-1] * math.log2(math.log2(self.nb_classes + 1))

        # Compute aleatoric
        pign_prob = np.zeros((result.shape[0], result.shape[1]))
        for k in range(result.shape[1]):
            pign_prob[:,k] = np.log2(result[:,k] + result[:,-1] / math.log2(self.nb_classes + 1))
        aleatoric = np.sum(-(result * pign_prob), axis=1)

        # Return uncertainties for input data
        return aleatoric, epistemic

    def _compute_bba(self, X, indices, distances):
        """
        Compute the bba for each element of X.

        Parameters
        -----
        X : ndarray
            Input array of X
        indices : ndarray
            Array of K nearest neighbors indices
        distances : ndarray
            Array of K nearest neighbors distances

        Returns
        -----
        bba : ndarray
            Array of bbas
        """
        # Initialisation of size and all bba
        n_samples = X.shape[0]
        bba = np.zeros((n_samples, self.nb_classes + 1))

        # Calculate a bba for each element of X
        for i in range(n_samples):
            m_list = np.zeros((self.n_neighbors, self.nb_classes + 1))

            # Construct a bba for each neighbors
            for j in range(self.n_neighbors):
                m = np.zeros(self.nb_classes + 1)
                m[-1] = 1

                for c in range(m.shape[0] - 2):
                    if isinstance(self.gamma, float):
                        weight = self.alpha * math.exp((-self.gamma) * (distances[i,j] ** self.beta)) * self.y_trained[int(indices[i,j]), c]
                    else:
                        weight = self.alpha * math.exp((-self.gamma[int(indices[i,j])]) * (distances[i,j] ** self.beta)) * self.y_trained[int(indices[i,j]), c]
                    m[c + 1] = weight
                    m[-1] -= weight

                m_list[j] = m
            
            # Compute normalized combination of bba
            m_normalized = np.array(ibelief.DST(m_list.T, 2))

            # Append the normalized bba to the array
            bba[i] = m_normalized.T
        
        return bba

    def _compute_distances(self, X):
        """
        Compute the euclidian distances with each neighbors.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        indices : ndarray
            Array of K nearest neighbors indices
        distances : ndarray
            Array of K nearest neighbors distances
        """

        # Fast find Nearest Neighbors
        index = faiss.IndexFlatL2(self.X_trained.shape[1]) 
        index.add(self.X_trained)
        D, I = index.search(X, self.n_neighbors)

        return I, np.sqrt(D)

    def _predict(self, X):
        """
        Compute distances and predicted bba on the input.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        result : ndarray
            Array of normalized bba
        """
        
        # Compute distances with k nearest neighbors
        neighbours_indices, neighbours_distances = self._compute_distances(X)

        # Compute bba
        result = self._compute_bba(X, neighbours_indices, neighbours_distances)

        return result

    def _init_parameters(self, alpha=ALPHA, beta=BETA):
        # Init alpha and beta
        self.alpha = alpha
        self.beta = beta

        # Init parameter gamma
        self.gamma = self._compute_gamma()

    def _compute_gamma(self, unique_gamma=True):
        """
        Compute unique gamma parameter. .

        Returns
        -----
        gamma : int
            Value of gamma
        """

        indices = np.random.choice(self.size, (min(10000,int(self.size / 2)), 2), replace=False)
        distances = np.linalg.norm(self.X_trained[indices[:, 0]] - self.X_trained[indices[:, 1]], axis=1)
        mean_distance = np.nanmean(distances)
        gamma =  1 / (mean_distance  ** self.beta)
        return gamma
