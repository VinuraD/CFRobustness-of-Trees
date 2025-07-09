# file: data_module.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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
        self.label_encoders = {}  # Dictionary to store label encoders for each categorical column
        
        # These will be updated after preprocessing
        self.feature_types = self.original_feature_types.copy()
        self.feature_actions = self.original_feature_actions.copy()
    
    def get_data(self, fold: int, raw_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and test data for specified fold.
        All metadata rows are excluded from both train and test sets.
        
        Args:
            fold: Fold number (0 to n_splits-1)
            raw_data: If True, return raw data. If False, return preprocessed data.
            
        Returns:
            train_df, test_df: Pandas DataFrames (metadata-free)
            
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
        
        # Ensure no metadata - data is already cleaned in __init__
        print(f"Fold {fold} - Train: {train_fold.shape[0]} samples, Test: {test_fold.shape[0]} samples")
        
        if raw_data:
            # Reset metadata to original when returning raw data
            self.feature_types = self.original_feature_types.copy()
            self.feature_actions = self.original_feature_actions.copy()
            # Verify no missing values in raw data
            if train_fold.isnull().any().any() or test_fold.isnull().any().any():
                print("Warning: Found missing values in raw data - this should not happen")
            return train_fold, test_fold
        else:
            # Fit preprocessors on train data and apply to both train and test
            self._fit_preprocessors(train_fold)
            train_processed = self._preprocess_data(train_fold)
            test_processed = self._preprocess_data(test_fold)
            # Verify no missing values in processed data
            if train_processed.isnull().any().any() or test_processed.isnull().any().any():
                print("Warning: Found missing values in processed data - this should not happen")
            return train_processed, test_processed
    
    def print_fold_summary(self):
        """Print a summary of all folds."""
        print(f"\nDataset Summary:")
        print(f"Total samples (after cleaning): {len(self.df)}")
        print(f"Number of features: {len(self.df.columns) - 1}")  # -1 for label column
        print(f"Label column: {self.label_col}")
        print(f"Number of folds: {self.n_splits}")
        print("\nFold Summary:")
        print("-" * 50)
        for fold in range(self.n_splits):
            train_idx, test_idx = self.fold_indices[fold]
            print(f"Fold {fold}: Train={len(train_idx)} samples, Test={len(test_idx)} samples")
        print("-" * 50)
    
    def get_metadata(self) -> Dict[str, any]:
        """Get metadata for counterfactual generation."""
        return {
            'feature_types': self.feature_types.copy(),
            'feature_actions': self.feature_actions.copy(),
            'label_column': self.label_col
        }
    
    def _clean_data(self):
        """Clean data and fix types. Drop rows with any missing values."""
        print(f"Original data shape (after removing metadata): {self.df.shape}")
        
        # Handle missing value sentinels for all columns first
        for col in self.df.columns:
            # Replace missing value sentinels with NaN
            self.df[col] = self.df[col].replace(
                [-9, -8, -7, '-9', '-8', '-7', ' -9 ', ' -8 ', ' -7 ', 'nan', 'NaN', ''], 
                np.nan
            )
        
        # Drop rows with any missing values
        initial_rows = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        final_rows = len(self.df)
        print(f"Dropped {initial_rows - final_rows} rows with missing values")
        print(f"Final data shape: {self.df.shape}")
        
        # Now convert data types for remaining clean data
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
        
        # Handle feature columns
        for col, dtype in self.original_feature_types.items():
            if col == self.label_col:
                continue  # Skip target column - already handled
            
            # Convert to appropriate types
            if dtype in ['N', 'D', 'B']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                # For discrete/binary features, round but keep as float
                if dtype in ['D', 'B']:
                    self.df[col] = self.df[col].round()
                    # For binary features, ensure values are in {0, 1}
                    if dtype == 'B':
                        self.df[col] = self.df[col].clip(0, 1)
            else:  # Categorical
                self.df[col] = self.df[col].astype(str)
        
        # Final check - drop any rows that still have NaN after type conversion
        if self.df.isnull().any().any():
            before_final_clean = len(self.df)
            self.df = self.df.dropna().reset_index(drop=True)
            after_final_clean = len(self.df)
            if before_final_clean != after_final_clean:
                print(f"Dropped additional {before_final_clean - after_final_clean} rows after type conversion")
                print(f"Final cleaned data shape: {self.df.shape}")
    
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
        
        # Fit label encoders on categorical columns
        self.label_encoders = {}
        if categorical_cols:
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(train_data[col].astype(str))
                self.label_encoders[col] = le
    
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
        
        # Label encode categorical features
        if categorical_cols and self.label_encoders:
            for col in categorical_cols:
                if col in self.label_encoders:
                    # Convert to string first to ensure consistency
                    col_data = df[col].astype(str)
                    
                    # Handle unseen categories by using the most frequent class
                    le = self.label_encoders[col]
                    known_classes = set(le.classes_)
                    unknown_mask = ~col_data.isin(known_classes)
                    
                    if unknown_mask.any():
                        # Replace unknown categories with the most frequent known category
                        most_frequent = le.classes_[0]  # First class is usually most frequent after fit
                        col_data.loc[unknown_mask] = most_frequent
                        print(f"Warning: Found {unknown_mask.sum()} unknown categories in {col}, replaced with '{most_frequent}'")
                    
                    # Apply label encoding
                    result_df[col] = le.transform(col_data)
                    
                    # Update metadata - label encoded categoricals become discrete
                    self.feature_types[col] = 'D'  # discrete integers
                    # Keep original feature action
                    self.feature_actions[col] = self.original_feature_actions[col]
        
        return result_df


# Simple usage example
if __name__ == "__main__":
    print("=" * 60)
    print("DATA MODULE TEST - MISSING DATA HANDLING")
    print("=" * 60)
    
    # Load dataset
    dm = DataModule("../data/German-Credit.csv", n_splits=5, random_state=42)
    
    # Print fold summary
    dm.print_fold_summary()
    
    print(f"\n{'='*60}")
    print("TESTING ALL FOLDS - RAW DATA")
    print("="*60)
    
    # Test all folds with raw data
    for fold in range(dm.n_splits):
        print(f"\n--- Testing Fold {fold} ---")
        train_raw, test_raw = dm.get_data(fold=fold, raw_data=True)
        
        # Verify no missing values
        train_missing = train_raw.isnull().sum().sum()
        test_missing = test_raw.isnull().sum().sum()
        print(f"Missing values in train: {train_missing}")
        print(f"Missing values in test: {test_missing}")
        print(f"Shape - Train: {train_raw.shape}, Test: {test_raw.shape}")
    
    print(f"\n{'='*60}")
    print("TESTING FOLD 0 - PROCESSED DATA (LABEL ENCODING)")
    print("="*60)
    
    # Test processed data for fold 0
    train_processed, test_processed = dm.get_data(fold=0, raw_data=False)
    print(f"Processed data shapes - Train: {train_processed.shape}, Test: {test_processed.shape}")
    
    # Verify no missing values in processed data
    train_proc_missing = train_processed.isnull().sum().sum()
    test_proc_missing = test_processed.isnull().sum().sum()
    print(f"Missing values in processed train: {train_proc_missing}")
    print(f"Missing values in processed test: {test_proc_missing}")
    
    # Show metadata
    metadata = dm.get_metadata()
    print(f"\nLabel column: {metadata['label_column']}")
    print(f"Total features: {len(metadata['feature_types'])}")
    print(f"Feature actions: {set(metadata['feature_actions'].values())}")
    
    # Show changes from raw to processed
    print(f"\nFeature type changes (raw vs processed):")
    for col in dm.df.columns:
        if col != dm.label_col:
            original_type = dm.original_feature_types[col]
            processed_type = metadata['feature_types'][col]
            if original_type != processed_type:
                print(f"  {col}: {original_type} → {processed_type} (label encoded)")
            else:
                print(f"  {col}: {original_type} (scaled)" if original_type in ['N', 'D', 'B'] else f"  {col}: {original_type}")
    
    print(f"\n{'='*60}")
    print("DATA MODULE TEST COMPLETED - LABEL ENCODING")
    print("="*60)
