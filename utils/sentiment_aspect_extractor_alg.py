import pandas as pd
import spacy
#import aspect_extractor as ae
import swifter
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
candidate_labels = ['positive', 'negative', 'neutral']
nlp = spacy.load("en_core_web_sm")

pos_dep_nn= ['NN','NNP','NNS']


def get_dependency_pairs(text):
    doc = nlp(text)
    # get token list
    tok_l = doc.to_json()['tokens']
    # traverse parse tree and get dependency pairs
    dependencies = []
    for t in tok_l:
        head = tok_l[t['head']]
        #print(f"'{text[t['start']:t['end']]}' is {t['dep']} of '{text[head['start']:head['end']]}' which is a {t['tag']}")
        dependencies.append((f"{text[t['start']:t['end']]}", f"{t['dep']}", f"'{text[head['start']:head['end']]}'", f"{head['tag']}", f"{t['tag']}"))

    return dependencies

def extract_candidates(dependencies):
    final_dep_list = []
    for dep in dependencies:
        if (dep[1] == 'amod' and dep[-1] in pos_dep_nn) or (dep[1] == 'acomp' and dep[-1]=='nsubj'):
           final_dep_list.append(dep)

    return final_dep_list

def filter_candidates(reviews, dep_list):
    ## get keyword list
    kw_list = ae.get_final_kw_list(reviews)
    final_pairs = []
    for dep in dep_list:
        if dep[-2].strip('\'') in kw_list:
           ## check the sentiment
           final_pairs.append(','.join([dep[0], dep[-2].strip('\'')]))
    return final_pairs


def get_aspect_Sentiment_prompts(reviews):
    dependencies = get_dependency_pairs(reviews)
    aspec_sentiment_candidates = extract_candidates(dependencies)
    print(aspec_sentiment_candidates)
    final_dep_list = filter_candidates(reviews, aspec_sentiment_candidates)
    return ' | '.join(final_dep_list)

def filter_aspect_sentiment(aspect_sentiment):
    labels = classifier(aspect_sentiment, candidate_labels, device=0)
    if (labels['labels'][0] == 'positive' or labels['labels'][0] == 'negative') and labels['scores'][0] >= 0.8:
        return True
    else:
        return False


#df_train_df = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_df.csv')
#df_valid_df = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_valid_df.csv')
#df_test_df = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_test_df.csv')

### adding prompts

#df_train_df['prompted_reviews'] = df_train_df.swifter.apply(lambda row: ' | '.join([get_aspect_Sentiment_prompts(row['review']), 'reviews: '+ row['review']]), axis=1)
#df_valid_df['prompted_reviews'] = df_valid_df.swifter.apply(lambda row: ' | '.join([get_aspect_Sentiment_prompts(row['review']),'reviews: '+row['review']]), axis=1)
#df_test_df['prompted_reviews'] = df_test_df.swifter.apply(lambda row: ' | '.join([get_aspect_Sentiment_prompts(row['review']),'reviews: '+row['review']]), axis=1)

#df_train_df.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_prompted_aspect_sentiment.csv', index=False)
#df_valid_df.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_valid_prompted_aspect_sentiment.csv', index=False)
#df_test_df.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_test_prompted_aspect_sentiment.csv', index=False)

reviews= \
'''
Following the logic behind a program in mathlab . It is interesting to see it step by step Exam Doing the exam retake is very helpful to understand where I went wrong and how to improve/do better Finding out what i did correctly on the exam How computers actually work Contributions and thoughts of team mates on exam 2 How to calculate the Reynolds number using density , viscosity , diameter , and velocity . Exam retake . team work in exam If statement Retaking the exam and figuring out mistakes . Finishing the exam with my team . Learning how questions in the exam are answered Exam 2 retake I enjoyed pair programming with my partner ! SAE values were pretty darn interesting the exam recap was pretty useful to check our mistakes error function and Exam revisit with teammates , because I saw my errors . Team exam Understanding my mistakes on the exams was the most beneficial aspect to the exam in class . I feel that it would be much more of an engineering exercise to take the exams as a group , because it requires working through assignments in groups to achieve the goal set by supervisors in the work force Learn from my mistakes I made in the exam . Learning my mistakes from the exam last night . Nothing creating programs from flow charts and testing cases to ensure the charts work I like doing the flow charts . Creating the flowchart to determine the validity of a pipe I found interesting the use of flow charts and using flow charts to make decisions . I also found the interpretation of flow charts into matlab code interesting The most interesting in class was to learn how to use a flowchart . I learned that I made a lot of mistakes in the exam 2 . I enjoyed getting to review the exam with my teammates and discover what i ded wrong . understanding the problems I missed on the exam What I found most interesting wasworking with the flowchart and seeing how they actually relate to coding . Using flow charts in Matlab The fact that we could work together to do the flowcharts . I think doing flowcharts and logic stuff is much more interesting than what we 've been doing in the past . Today , I worked with my team to get our team exam completed . It went well since we knew what was expected of the exam . nothing was interesting today . I like converting the flowchart to Matlab code . My grade on the teaming reflection . Being able to translate the flowchart into matlab coding . the exam was dope . and Hyder was on time Doing exam with teammates test retake was enlightening the affect of an outlier on the r ^ 2 value is pretty interesting because it makes a big difference Nothing I really loved taking the test today The exam Nothing flowcharts are somewhat interesting flow charts make sense for the most part The usage of if-else statements was interesting Flow charts and selection structures Exam work went alright . flow charts make sense for now . Constructing flowchart pair working Team exam Topic of the problem 1 udf Doing the exam Turning a flowchart into a MATLAB function . Flowcharts The exam is hard I did n't like it Retaking the test and finding out everything I did wrong I found it interesting that an outlier of a data set can have such drastic effect on the r ^ 2 value I really liked getting to go over the exam with the whole group Going over exam I found the exam retake with my team to be interesting and helpful . Mapping out a while loop That you can code with less work and get the same results I enjoyed talking with my group about what we each got wrong on the exam
'''

#kw, kw_df = ae.get_final_kw_list(reviews)

#print(kw_df.head(10))

#print(kw)
dependencies = get_dependency_pairs(reviews)
print(dependencies)
#aspec_sentiment_candidates = extract_candidates(dependencies)

#print(aspec_sentiment_candidates)

print(get_aspect_Sentiment_prompts(reviews))
