import pandas as pd
from langdetect import detect
import swifter

course_reviews = '~/PhD/datasets/course_reviews/reviews_by_course.csv'

def get_lang_id(review):
    try:
        return detect(review)
    except:
        return ''
reviews_df = pd.read_csv(course_reviews)
reviews_df['lang_id'] = reviews_df.swifter.apply(lambda row: get_lang_id(row['Review']), axis=1)
reviews_df = reviews_df[reviews_df['lang_id'] == 'en']

group_ids, reviews =[],[]
for group_id, rev_df in reviews_df.groupby(by=['CourseId']):
    group_ids.append(group_id)
    reviews.append(' '.join(rev_df['Review'].tolist()))

pd.DataFrame(data={'course_id': group_ids, 'reviews': reviews}).to_csv('~/PhD/datasets/course_reviews/reviews_aggregated_coursera.csv', index=False)
#reviews_df.to_csv('~/PhD/datasets/course_reviews/reviews_by_course_langid.csv', index=False)
