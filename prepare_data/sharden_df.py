import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', action='store', dest='data',
                    help='train csv')
parser.add_argument('--out_dir', action='store', dest='out',
                    help='valid csv')
parser.add_argument('--split_size', action='store', dest='split_size',
                    help='yaml configuration file')
args = parser.parse_args()

##########################

print('----- Created Chunks ------')
df_data = pd.read_csv(args.data)
N,_ = df_data.shape
ix = int(float(args.split_size)*N)
end_ix = ix
start_ix = 0
chunk_counter = 1
while end_ix <= N:
    df_data.iloc[start_ix:min(N,end_ix),:].to_csv(os.path.join(args.out,''.join([args.data.strip('.csv'),'_chunk',str(chunk_counter),'.csv'])), index=False)
    start_ix += ix
    end_ix = end_ix+ix
    chunk_counter +=1

print('---- Done Creating Chunks ------')
