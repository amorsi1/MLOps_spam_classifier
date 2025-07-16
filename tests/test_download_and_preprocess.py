import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from MLOps_spam_classifier.src.model_train import load_data
from MLOps_spam_classifier.src.download_and_preprocess import embed_df_text, csv_preprocessing

def test_embeddings():

    # Load the combined dataset
    comb_df = load_data('data/processed/combined_spam_ham_dataset.pkl')
    
    # Check if embeddings are present
    assert 'embedding' in comb_df.columns, "Embeddings column not found in DataFrame"

    row_index = 47
    read_row = comb_df.iloc[row_index,:]['embedding']
    output = embed_df_text(comb_df.iloc[row_index:row_index+2,:])
    print(output['embedding'])
    output_row = output.iloc[0]['embedding']
    # check if embedding corresponds
    np.testing.assert_allclose(read_row, output_row, rtol=1e-4) , f"Embedding mismatch for row {row_index}"
    print("Embedding test passed! embedding matches saved df!")



def test_csv_preprocessing():
    df = pd.read_csv('data/raw/CEAS_08.csv')
    processed_df = csv_preprocessing(df, 'CEAS_08.csv')
    # Check if columns are dropped
    assert len(df.columns) > len(processed_df.columns), "Columns were not dropped" #this test leaves room to drop some columns but not others
    assert 'source' in processed_df.columns, "Source column not added"
    print("CSV preprocessing test passed!")

