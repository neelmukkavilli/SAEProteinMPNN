
import pickle
import pandas as pd

csv_outputs = []
for i in range(3):
    csv_outputs.append(f'created_data/encodings/output_' + 'validlosstest' + '_' + str(i) + '.pkl')

for file in csv_outputs:
    dfs = []
    with open(file, "rb") as f:
        while True:
            try:
                dfs.append(pickle.load(f))
            except EOFError:
                break

    full_df = pd.concat(dfs, ignore_index=True)
    print(full_df)
