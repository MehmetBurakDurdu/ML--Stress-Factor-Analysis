import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class StressLevelModel:
    def __init__(self, data_path):
        """
        Initialize the StressLevelModel.

        Parameters:
        - data_path (str): The path to the CSV file containing the stress level dataset.
        """
        self.df = pd.read_csv(data_path)
        self.linear_cols = ['anxiety_level', 'mental_health_history', 'depression',
                            'headache', 'blood_pressure', 'breathing_problem', 'noise_level', 'living_conditions',
                            'safety', 'basic_needs', 'academic_performance', 'study_load',
                            'teacher_student_relationship', 'future_career_concerns', 'social_support',
                            'peer_pressure', 'extracurricular_activities', 'bullying']
        self.logistic_cols = ['stress_level', 'sleep_quality']

    def remove_unnamed_column(self):
        """
        Remove the 'Unnamed: 0' column if present in the dataframe.
        """
        if "Unnamed: 0" in self.df.columns:
            del self.df["Unnamed: 0"]

    def fill_missing_values(self):
        """
        Fill missing values in the dataframe with mean values.
        """
        self.df.fillna(self.df.mean(), inplace=True)

    def add_correlation_heatmap(self):
            """
            Add a correlation heatmap to the visualization.
            """
            plt.figure(figsize=(8, 6))
            correlation_matrix = self.df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.show()

    def find_column_with_lowest_correlation(self):
        """
        Find the column with the lowest correlation coefficient to the target variable.

        Returns:
        - min_corr_col (str): The column with the lowest correlation coefficient.
        """
        correlation_matrix = self.df.corr()

        # Find the column with the correlation coefficient closest to 0
        min_corr_col = correlation_matrix.iloc[:-1, -1].idxmin()

        # Print the column with the lowest correlation coefficient
        print(f"Column with the lowest correlation to the target variable: {min_corr_col}")

        return min_corr_col


    def drop_column_with_lowest_correlation(self):
        """
        Drop the column with the lowest correlation coefficient to the target variable.
        """
        min_corr_col = self.find_column_with_lowest_correlation()
        
        # Print the column about to be dropped
        print(f"About to drop column with closest correlation to 0: {min_corr_col}")

        # Drop the column with the closest correlation to 0
        self.df.drop(columns=min_corr_col, inplace=True)
        print(f"Dropped column with closest correlation to 0: {min_corr_col}")

        # Update linear columns after dropping a column
        self.update_linear_cols(min_corr_col)


    def update_linear_cols(self, dropped_column):
        """
        Update the linear columns list after dropping a column.
        """
        self.linear_cols = [col for col in self.linear_cols if col != dropped_column]

    def split_data_linear(self):
        """
        Split the data into linear regression training and testing sets.
        """
        X_linear = self.df[self.linear_cols]
        y_linear = self.df['stress_level']
        return train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

    def split_data_logistic(self, target_class=0):
        """
        Split the data into logistic regression training and testing sets.

        Parameters:
        - target_class (int): The class to be predicted against the rest (default is 0).
        """
        X_logistic = self.df[self.logistic_cols]
        y_logistic = (self.df['sleep_quality'] == target_class).astype(int)
        return train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

    def train_linear_model(self, X_train, y_train):
        """
        Train a linear regression model.

        Parameters:
        - X_train (DataFrame): The features for training.
        - y_train (Series): The target variable for training.

        Returns:
        - model: The trained linear regression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_logistic_model(self, X_train, y_train):
        """
        Train a logistic regression model.

        Parameters:
        - X_train (DataFrame): The features for training.
        - y_train (Series): The target variable for training.

        Returns:
        - model: The trained logistic regression model.
        """
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_linear_model(self, model, X_test, y_test):
        """
        Evaluate a linear regression model.

        Parameters:
        - model: The trained linear regression model.
        - X_test (DataFrame): The features for testing.
        - y_test (Series): The target variable for testing.
        """
        y_pred = model.predict(X_test)
        rmse = self.calculate_rmse(y_test, y_pred)
        self.visualize_linear_predictions(y_test, y_pred)

    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate the root mean squared error (RMSE) between true and predicted values.

        Parameters:
        - y_true (Series): The true target variable values.
        - y_pred (array): The predicted target variable values.

        Returns:
        - rmse: The calculated RMSE.
        """
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print('Linear Regression RMSE:', rmse)
        return rmse

    def visualize_linear_predictions(self, y_true, y_pred):
        """
        Visualize linear regression predictions.

        Parameters:
        - y_true (Series): The true target variable values.
        - y_pred (array): The predicted target variable values.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', lw=2)
        plt.title('Linear Regression: Actual vs Predicted')
        plt.xlabel('Actual Stress Level')
        plt.ylabel('Predicted Stress Level')
        plt.grid(True)
        plt.show()

    def evaluate_logistic_model(self, model, X_test, y_test):
        """
        Evaluate a logistic regression model.

        Parameters:
        - model: The trained logistic regression model.
        - X_test (DataFrame): The features for testing.
        - y_test (Series): The true target variable for testing.
        """
        y_pred = model.predict(X_test)
        accuracy, conf_matrix, classification_rep = self.calculate_classification_metrics(y_test, y_pred)
        self.print_classification_metrics(accuracy, conf_matrix, classification_rep)
        self.visualize_roc_curve(model, X_test, y_test)

    def calculate_classification_metrics(self, y_true, y_pred):
        """
        Calculate classification metrics (accuracy, confusion matrix, classification report).

        Parameters:
        - y_true (Series): The true target variable values.
        - y_pred (array): The predicted target variable values.

        Returns:
        - accuracy: The accuracy of the model.
        - conf_matrix: The confusion matrix.
        - classification_rep: The classification report.
        """
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred)
        return accuracy, conf_matrix, classification_rep

    def print_classification_metrics(self, accuracy, conf_matrix, classification_rep):
        """
        Print classification metrics.

        Parameters:
        - accuracy: The accuracy of the model.
        - conf_matrix: The confusion matrix.
        - classification_rep: The classification report.
        """
        print('Logistic Regression Classification Model:')
        print('Accuracy:', accuracy)
        print('Confusion Matrix:')
        print(conf_matrix)
        print('Classification Report:')
        print(classification_rep)

    def visualize_roc_curve(self, model, X_test, y_test):
        """
        Visualize the ROC curve for a logistic regression model.

        Parameters:
        - model: The trained logistic regression model.
        - X_test (DataFrame): The features for testing.
        - y_test (Series): The true target variable for testing.
        """
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def visualize_data(self):
        """
        Visualize the data using subplots.
        """
        r = 4  # Subplot grid row count
        c = 6  # Subplot grid column count
        it = 1  # Iterator for subplot position

        plt.figure(figsize=(18, 12))  # Set the overall figure size

        for i in self.df.columns:
            plt.subplot(r, c, it)
            if self.df[i].nunique() > 6:
                sns.kdeplot(self.df[i])
                plt.grid()
            else:
                sns.countplot(x=self.df[i])
            plt.xlabel(i)
            plt.ylabel('Count' if self.df[i].nunique() <= 6 else 'Density')
            it += 1

        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Run the complete workflow:
        - Visualize the data
        - Add correlation heatmap
        - Preprocess the data
        - Train and evaluate linear regression model
        - Train and evaluate logistic regression model
        - Visualize the relationship between stress and sleep quality
        """
        # Visualize the data
        self.visualize_data()

        # Add correlation heatmap
        self.add_correlation_heatmap()

        # Preprocess the data
        self.remove_unnamed_column()
        self.fill_missing_values()
        self.drop_column_with_lowest_correlation()

        # Linear Regression
        X_linear_train, X_linear_test, y_linear_train, y_linear_test = self.split_data_linear()
        linear_model = self.train_linear_model(X_linear_train, y_linear_train)
        self.evaluate_linear_model(linear_model, X_linear_test, y_linear_test)

        # Logistic Regression
        target_class = 1  # Choose the class you want to predict against the rest
        X_logistic_train, X_logistic_test, y_logistic_train, y_logistic_test = self.split_data_logistic(target_class)
        logistic_model = self.train_logistic_model(X_logistic_train, y_logistic_train)
        self.evaluate_logistic_model(logistic_model, X_logistic_test, y_logistic_test)

        # Visualize the relationship between stress and sleep quality
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            y_logistic_test,
            logistic_model.predict_proba(X_logistic_test)[:, 1],
            c=X_logistic_test['sleep_quality'],  # Use sleep quality as color gradient
            cmap='viridis',  # Choose a color map
            alpha=0.8,
        )
        plt.colorbar(scatter, label='Stress level')  # Add color bar for sleep quality
        plt.plot([0, 1], [1, 0], color='red', linestyle='--', lw=2)  # Red dashed line for reference
        plt.title('Logistic Regression: Actual Sleep Quality vs. Predicted Stress Level')
        plt.xlabel('Actual Sleep Quality (Class 1)')
        plt.ylabel('Predicted Stress Level')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    data_path = 'StressLevelDataset.csv'  # Adjust the file path if needed
    stress_model = StressLevelModel(data_path)
    stress_model.run()

