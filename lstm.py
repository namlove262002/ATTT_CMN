# Import required libraries
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint

import itertools

# Load uploaded file
class MachineLearning():

    def __init__(self):
        print("Loading dataset ...")
        # Update this line to read the uploaded file
        # Upload CSV file
        file_path1 = 'FlowStatsfile.csv'
        file_path2 = 'FlowStatsfile_DDoS.csv'
        # Load the first CSV file (limit to the first 100,000 rows)
        df1 = pd.read_csv(file_path1, nrows=50000)

        # Load the second CSV file (limit to the first 100,000 rows)
        df2 = pd.read_csv(file_path2, nrows=50000)

        # Merge the two DataFrames
        self.flow_dataset = pd.concat([df1, df2], ignore_index=True)

        # self.flow_dataset = pd.read_csv(file_path)
        self.flow_dataset= self.flow_dataset.drop(['timestamp', 'datapath_id','flow_id'], axis=1)
        print("Columns after dropping:", self.flow_dataset.columns)  # Verify columns
        # Preprocess data by removing '.' characters
        self.flow_dataset.iloc[:, 0] = self.flow_dataset.iloc[:, 0].str.replace('.', '')
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        
    def flow_training(self,epochs):
        print("Flow Training ...")

        # Extract features and labels
        X_flow = self.flow_dataset.iloc[:, :-1].values.astype('float64')
        y_flow = self.flow_dataset.iloc[:, -1].values

        # Standardize the features
        scaler = StandardScaler()
        X_flow = scaler.fit_transform(X_flow)

        # Reshape for LSTM (samples, timesteps, features)
        X_flow = X_flow.reshape((X_flow.shape[0], X_flow.shape[1], 1))

        # Split data into train and test sets
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        # Define LSTM model
        print("Building LSTM model ...")
        model = Sequential()
        model.add(LSTM(80, return_sequences=False, input_shape=(X_flow_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.add(Activation('sigmoid'))

        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Train model
        history = model.fit(X_flow_train, y_flow_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1,callbacks=[checkpoint])

        # Predictions
        y_flow_pred = (model.predict(X_flow_test) > 0.5).astype("int32")

        # Evaluate model
        print("------------------------------------------------------------------------------")
        print("Confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)
        print("Success accuracy = {0:.2f} %".format(acc*100))
        print("Precision score = {0:.2f} %".format(precision_score(y_flow_test, y_flow_pred)*100))
        print("Recall score = {0:.2f} %".format(recall_score(y_flow_test, y_flow_pred)*100))
        print("F1 score = {0:.2f} %".format(f1_score(y_flow_test, y_flow_pred)*100))
        print("AUC score = {0:.2f} %".format(roc_auc_score(y_flow_test, y_flow_pred)*100))

        # Plot confusion matrix
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Normal', 'DDoS'], rotation=45)
        plt.yticks(tick_marks, ['Normal', 'DDoS'])
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        # Plot accuracy and loss
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_flow_test, y_flow_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        print("------------------------------------------------------------------------------")


        model.save('trained_lstm_model.h5')
        print("Model saved as 'trained_lstm_model.h5'.")
        
def main():
    start = datetime.now()
    ml = MachineLearning()
    ml.flow_training(epochs=50)
    end = datetime.now()
    print("Training time: ", (end-start))

if __name__ == "__main__":
    main()


