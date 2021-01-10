from finbert_embedding.embedding import FinbertEmbedding
import pandas as pd

finbert = FinbertEmbedding()

df = pd.read_csv('../datasets/headlines_clean.csv', parse_dates=['date'])
df['embedding'] = 0

def foo(row):
    row['embedding'] = finbert.sentence_vector(row['text']).numpy()
    return row

print('Beginning........')
df = df.apply(foo, axis=1)
print(df)

df = pd.concat([df['date'], pd.DataFrame(df['embedding'].values.tolist())], axis=1)
print(df)

df.to_csv('../datasets/headline_embeddings.csv', index=False)
