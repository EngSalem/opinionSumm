import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--predicted', action='store', dest='predicted',
                    help='')
parser.add_argument('--nn_file', action='store', dest='nn_file',
                    help='')
parser.add_argument('--train', action='store', dest='train',
                    help='')

parser.add_argument('--combined', action='store', dest='combined',
                    help='')

args = parser.parse_args()

df_train = pd.read_csv(args.train)
df_nn = pd.read_csv(args.nn_file).rename(columns={'input':'review'})
df_predicted = pd.read_csv(args.predicted).rename(columns={'generated_summary':'summary'})

pd.concat([df_train, pd.DataFrame(data={'review': df_nn['reviews'].tolist(), 'summary': df_predicted['summary'].tolist()})]).to_csv(args.combined)





