from collections import Counter
import numpy as np

class KNeighborsClassifier:

    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        """
        Params :
            self : object of class KNeighborsClassifier
                object calling the function
            X_train : numpy array 
                All the training data
            y_train : numpy array
                All the training labels
        """
        self.data=X_train
        self.labels=y_train

    def compute_distances(self,point):
        """
        Params :
            self : object of class KNeighborsClassifier
                object calling the function
            point : list
                a single point
        returns:
            distances : numpy array
                distance of point from all other points
        """
        distances=[]
        for train_point in self.data:
            distances.append(np.linalg.norm(point-train_point))
        return np.array(distances)

    def get_neighbours(self,point):
        """
        Params :
            self : object of class KNeighborsClassifier
                object calling the function
            point : numpy array
                represents a point
        Returns :
            indices : k X 1 numpy array
                indices of 'k' nearest points
        """
        # first get the ditance from point to all other points
        distances=self.compute_distances(point)
        # sort it
        key_value={}
        keys=range(len(distances))
        for i in keys:
            key_value[i]=distances[i]
        
        sorted_key_value = sorted(key_value.items(), key=lambda kv: kv[1])
        sorted_key_value=np.array([list(item) for item in sorted_key_value])

        # return top k indices
        return sorted_key_value[:self.k,0]

    def output(self,neighbour_indixes):
        """
        Params :
            self : object of class KNeighborsClassifier
                object calling the function
            neighbour_indixes : numpy array
                represents all 'k' points that are neighbours of a given point
        Returns :
            label : int
                class of the object as obtained by the majority of neighbour classes
        """
        frequency_dict = Counter(neighbour_indixes)
        most_common_index=np.array([list(item) for item in frequency_dict.most_common(1)])
        most_common_index=most_common_index[:,0]
        return self.labels[int(most_common_index[0])]

    def predict(self, X_test):
        """
        Params :
            self : object of class KNeighborsClassifier
                object calling the function
            X_test : numpy array
                test data points 
        Returns :
            predictions : numpy array
                class of the test data points as obtained by KNN model
        """
        predictions=[]
        for x in X_test:
            neighbours=self.get_neighbours(x)
            output=self.output(neighbours)
            predictions.append(output)
        return np.array(predictions)
        # raise NotImplementedError()

    def score(self, y_pred_test, y_test):
        """
        Params :
            self : object of class KNeighborsClassifier
                object calling the function
            y_pred_test : numpy array
                KNN predicted labels
            y_test : numpy array
                test data labels 
        Returns :
            accuracy : Float
                accuracy of predictions of KNN model
        """
        num=np.sum(y_pred_test==y_test)
        den=len(y_test)
        return num/den
