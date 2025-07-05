 file: data_module.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

class DataModule:
    """
    Simple data module for robustness testing.
    - K-fold CV on entire dataset (excluding metadata)
    - Each fold gives train/test split
    - Raw data or preprocessed data option
    """

    def __init__(self, csv_path: str, n_splits: int = 5, random_state: int = 42):
        self.csv_path = Path(csv_path)
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Load and parse metadata
        raw = pd.read_csv(self.csv_path)
        self.original_feature_types = raw.iloc[0].to_dict()
        self.original_feature_actions = raw.iloc[1].to_dict()
        
        # Get actual data (skip metadata rows)
        self.df = raw.drop([0, 1]).reset_index(drop=True)
        
        # Find label column
        self.label_col = [col for col, tag in self.original_feature_actions.items() if tag == "PREDICT"][0]
        
        # Clean data and fix types
        self._clean_data()
        
        # Create k-fold splits on entire dataset
        self.kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.fold_indices = list(self.kfold.split(self.df.drop(columns=[self.label_col]), 
                                                  self.df[self.label_col]))
        
        # Store preprocessing objects (will be fitted for each fold)
        # Note: These get overwritten each time get_data() is called with raw_data=False
        self.scaler = None
        self.encoder = None
        
        # These will be updated after preprocessing
        self.feature_types = self.original_feature_types.copy()
        self.feature_actions = self.original_feature_actions.copy()
    
    def get_data(self, fold: int, raw_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and test data for specified fold.
        
        Args:
            fold: Fold number (0 to n_splits-1)
            raw_data: If True, return raw data. If False, return preprocessed data.
            
        Returns:
            train_df, test_df: Pandas DataFrames
            
        Note:
            When raw_data=False, preprocessing objects (scaler/encoder) are fitted on 
            this fold's training data and will overwrite any previous fitted objects.
        """
        if not 0 <= fold < self.n_splits:
            raise ValueError(f"Fold must be between 0 and {self.n_splits-1}")
        
        # Get train/test indices for this fold
        train_idx, test_idx = self.fold_indices[fold]
        train_fold = self.df.iloc[train_idx].reset_index(drop=True)
        test_fold = self.df.iloc[test_idx].reset_index(drop=True)
        
        if raw_data:
            # Reset metadata to original when returning raw data
            self.feature_types = self.original_feature_types.copy()
            self.feature_actions = self.original_feature_actions.copy()
            return train_fold, test_fold
        else:
            # Fit preprocessors on train data and apply to both train and test
            self._fit_preprocessors(train_fold)
            train_processed = self._preprocess_data(train_fold)
            test_processed = self._preprocess_data(test_fold)
            return train_processed, test_processed
    
    def get_metadata(self) -> Dict[str, any]:
        """Get metadata for counterfactual generation."""
        return {
            'feature_types': self.feature_types.copy(),
            'feature_actions': self.feature_actions.copy(),
            'label_column': self.label_col
        }
    
    def _clean_data(self):
        """Clean data and fix types."""
        # First, handle the label column - convert to numeric if possible
        try:
            # Try direct numeric conversion first
            self.df[self.label_col] = pd.to_numeric(self.df[self.label_col], errors='raise')
        except (ValueError, TypeError):
            # If it fails, use label encoding for string labels
            le = LabelEncoder()
            self.df[self.label_col] = le.fit_transform(self.df[self.label_col])
            # Store label encoder for potential future use
            self.label_encoder = le
        
        # Handle missing values and convert types for all feature columns
        for col, dtype in self.original_feature_types.items():
            if col == self.label_col:
                continue  # Skip target column - already handled
                
            # Handle missing value sentinels for all types (both numeric and string)
            self.df[col] = self.df[col].replace(
                [-9, -8, -7, '-9', '-8', '-7', ' -9 ', ' -8 ', ' -7 '], 
                np.nan
            )
            
            # Convert to numeric and handle missing values
            if dtype in ['N', 'D', 'B']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                # Fill missing values with median
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
                # For discrete/binary features, round but keep as float
                if dtype in ['D', 'B']:
                    self.df[col] = self.df[col].round()
                    # For binary features, ensure values are in {0, 1}
                    if dtype == 'B':
                        self.df[col] = self.df[col].clip(0, 1)
            else:  # Categorical
                self.df[col] = self.df[col].astype(str)
                # Fill missing values with mode
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col] = self.df[col].fillna(mode_val[0])
                else:
                    self.df[col] = self.df[col].fillna('Unknown')
    
    def _fit_preprocessors(self, train_data: pd.DataFrame):
        """Fit preprocessing objects on training data for current fold."""
        # Get feature columns (exclude target)
        feature_cols = [col for col in train_data.columns if col != self.label_col]
        
        # Separate numeric and categorical features
        numeric_cols = [col for col in feature_cols if self.original_feature_types[col] in ['N', 'D', 'B']]
        categorical_cols = [col for col in feature_cols if self.original_feature_types[col] == 'C']
        
        # Safety check for overlapping columns
        assert not (set(numeric_cols) & set(categorical_cols)), \
            f"Overlapping numeric and categorical columns: {set(numeric_cols) & set(categorical_cols)}"
        
        # Fit scaler on numeric columns
        if numeric_cols:
            self.scaler = MinMaxScaler()
            self.scaler.fit(train_data[numeric_cols])
        
        # Fit encoder on categorical columns - handle version compatibility
        if categorical_cols:
            # Try new parameter name first, fall back to old
            try:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:
                # Fallback for older scikit-learn versions
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.encoder.fit(train_data[categorical_cols])
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to data and update metadata."""
        result_df = df.copy()
        feature_cols = [col for col in df.columns if col != self.label_col]
        
        # Separate numeric and categorical features
        numeric_cols = [col for col in feature_cols if self.original_feature_types[col] in ['N', 'D', 'B']]
        categorical_cols = [col for col in feature_cols if self.original_feature_types[col] == 'C']
        
        # Scale numeric features
        if numeric_cols and self.scaler:
            result_df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # One-hot encode categorical features
        if categorical_cols and self.encoder:
            # Get encoded features
            encoded_features = self.encoder.transform(df[categorical_cols])
            
            # Get feature names - try new method first, fall back to old for compatibility
            try:
                encoded_columns = self.encoder.get_feature_names_out(categorical_cols)
            except AttributeError:
                # Fallback for older scikit-learn versions
                encoded_columns = self.encoder.get_feature_names(categorical_cols)
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=df.index)
            
            # Drop original categorical columns and add encoded ones
            result_df = result_df.drop(columns=categorical_cols)
            result_df = pd.concat([result_df, encoded_df], axis=1)
            
            # Update metadata to reflect the new structure
            for new_col in encoded_columns:
                self.feature_types[new_col] = 'B'  # one-hots are binary
                # Extract base column name and inherit its action
                base_col = new_col.split('_')[0] if '_' in new_col else new_col
                if base_col in self.original_feature_actions:
                    self.feature_actions[new_col] = self.original_feature_actions[base_col]
                else:
                    self.feature_actions[new_col] = 'FREE'  # default action
            
            # Remove original categorical columns from metadata
            for col in categorical_cols:
                self.feature_types.pop(col, None)
                self.feature_actions.pop(col, None)
        
        return result_df


# Simple usage example
if __name__ == "__main__":
    # Load dataset
    dm = DataModule("data/German-Credit.csv", n_splits=5, random_state=42)
    
    print("Dataset split sizes for each fold:")
    for fold in range(dm.n_splits):
        train_raw, test_raw = dm.get_data(fold=fold, raw_data=True)
        print(f"Fold {fold}: Train={train_raw.shape[0]}, Test={test_raw.shape[0]}")
    
    # Test with fold 0
    train_raw, test_raw = dm.get_data(fold=0, raw_data=True)
    print(f"\nFold 0 - Raw data: Train={train_raw.shape}, Test={test_raw.shape}")
    print(f"Raw metadata features: {len(dm.get_metadata()['feature_types'])}")
    
    train_processed, test_processed = dm.get_data(fold=0, raw_data=False)
    print(f"Fold 0 - Processed data: Train={train_processed.shape}, Test={test_processed.shape}")
    print(f"Processed metadata features: {len(dm.get_metadata()['feature_types'])}")
    
    # Show label column handling
    metadata = dm.get_metadata()
    print(f"\nLabel column: {metadata['label_column']}")
    print(f"Label type: {type(dm.df[dm.label_col].iloc[0])}")
    print(f"Total samples: {len(dm.df)}")
    
    # Test categorical encoding
    print(f"\nSample feature types after processing:")
    for col, ftype in list(dm.get_metadata()['feature_types'].items())[:5]:
        print(f"  {col}: {ftype}")
