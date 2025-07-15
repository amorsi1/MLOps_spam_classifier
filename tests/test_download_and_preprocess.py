import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from MLOps_spam_classifier.src.model_train import load_data
from MLOps_spam_classifier.src.download_and_preprocess import embed_text, csv_preprocessing

def test_embeddings():

    # Load the combined dataset
    comb_df = load_data('data/processed/combined_spam_ham_dataset.pkl')
    
    # Check if embeddings are present
    assert 'embedding' in comb_df.columns, "Embeddings column not found in DataFrame"

    row = comb_df.iloc[3,:]
    output = embed_text(row)
    # check if embedding corresopnds
    assert np.testing.assert_array_almost_equal(row['embedding'], output['embedding']) , "Embedding mismatch for row 3"
    print("Embedding" test passed!")

def test_csv_preprocessing():
    df = pd.read_csv('data/raw/CEAS_08.csv')
    processed_df = csv_preprocessing(df, 'CEAS_08.csv')
    # Check if columns are dropped
    assert len(df.columns) > len(processed_df.columns), "Columns were not dropped" #this test leaves room to drop some columns but not others
    assert 'source' in processed_df.columns, "Source column not added"
    print("CSV preprocessing test passed!")



if __name__ == "__main__":
    # test_embeddings()