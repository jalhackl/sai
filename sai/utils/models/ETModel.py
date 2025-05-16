from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os
from gaia.utils.models import MLModel

pd.options.mode.chained_assignment = None


class ETModel(MLModel):
    """
    An Extra Trees classifier model class for training and inference,
    designed for biological datasets such as those in genomics.
    This class provides static methods to train and use an Extra Trees classifier.
    """

    @staticmethod
    def train(
        training_data: str,
        model_file: str,
        seed: int = None,
        n_estimators: int = 100,
        max_depth: int = None,
        class_weight: str = None,
        is_scaled: bool = False,
    ) -> None:
        """
        Train an Extra Trees classifier model using the provided training data,
        and save the model and scaler (if used) to disk.

        Parameters
        ----------
        training_data : str
            Path to the training data file in tab-separated format.
        model_file : str
            Path where the trained model will be saved.
        seed : int, optional
            Random seed for reproducibility. Default: None.
        n_estimators : int, optional
            Number of trees in the forest. Default: 100.
        max_depth : int, optional
            The maximum depth of the tree. Default: None.
        class_weight : str, optional
            Weights associated with classes (e.g., "balanced"). Default: None.
        is_scaled : bool, optional
            Whether to scale the features. Default: False.
        """
        features = pd.read_csv(training_data, sep="\t")
        output_dir = os.path.dirname(model_file)
        os.makedirs(output_dir, exist_ok=True)

        labels = features["Label"]
        data = features.drop(
            columns=["Chromosome", "Start", "End", "Sample", "Replicate", "Label"]
        ).values

        if is_scaled:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            joblib.dump(scaler, f"{model_file}.scaler")

        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=seed,
        )
        model.fit(data, labels.astype(int))

        joblib.dump(model, model_file)


    @staticmethod
    def infer(
        inference_data: str, model_file: str, output_file: str, is_scaled: bool = False
    ) -> None:
        """
        Perform inference using a trained Extra Trees classifier on new data,
        outputting class probabilities to a specified file.

        Parameters
        ----------
        inference_data : str
            Path to the inference data file in tab-separated format.
        model_file : str
            Path to the saved trained model.
        output_file : str
            Path to save the inference results.
        is_scaled : bool, optional
            Whether to apply scaling using the saved scaler. Default: False.
        """
        features = pd.read_csv(inference_data, sep="\t")
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        data = features.drop(columns=["Chromosome", "Start", "End", "Sample"]).values

        if is_scaled:
            scaler = joblib.load(f"{model_file}.scaler")
            data = scaler.transform(data)

        model = joblib.load(model_file)

        predictions = model.predict_proba(data)
        prediction_df = features[["Chromosome", "Start", "End", "Sample"]]

        class_names = {
            "0": "Non_Intro",
            "1": "Intro",
        }

        classes = model.classes_
        for i in range(len(classes)):
            class_name = class_names[f"{classes[i]}"]
            prediction_df[f"{class_name}_Prob"] = predictions[:, i]

        prediction_df.sort_values(by=["Sample", "Chromosome", "Start", "End"]).to_csv(
            output_file, sep="\t", index=False
        )
