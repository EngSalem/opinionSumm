import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-courses_table', type=str)
parser.add_argument('-train_df', type=str)
parser.add_argument('-valid_df', type=str)
parser.add_argument('-test_df', type=str)
args = parser.parse_args()

train_df, valid_df, test_df = [], [], []
for _, _courseDF in pd.read_csv(args.courses_table).groupby(['course']):
    courseTrain, courseTest = train_test_split(_courseDF, test_size=0.2)
    courseValid , courseTest = train_test_split(_courseDF, test_size=0.5)
    train_df.append(courseTrain)
    valid_df.append(courseValid)
    test_df.append(courseTest)

#####
pd.concat(train_df).to_csv(args.train_df, index=False)
print('train size', pd.concat(train_df).shape[0])
pd.concat(valid_df).to_csv(args.valid_df, index=False)
print('valid size', pd.concat(valid_df).shape[0])
pd.concat(test_df).to_csv(args.test_df, index=False)
print('test size', pd.concat(test_df).shape[0])


