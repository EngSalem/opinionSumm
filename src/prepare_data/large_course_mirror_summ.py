import os
import json
import logging
import pandas as pd

DataDir = '/home/mohamed/PhD/courseMirror/annotation_per_annotator/'

annotators, courses, lecture_ids, prompts, reflectdfions, concepts, extractive_summ, abstractive_summ, phrase_summ = [], [], [], [], [], [], [], [],[]
for annotator in os.listdir(DataDir):
    for course in os.listdir(os.path.join(DataDir, annotator)):
        try:
            for lecture in os.listdir(os.path.join(DataDir, annotator, course, 'Done')):
                with open(os.path.join(DataDir, annotator, course, 'Done', lecture)) as f:
                    data_lecture = json.load(f)
                    ## basic course , annotator, and lecture information
                    annotators.append(annotator)
                    courses.append(course)
                    lecture_ids.append(data_lecture['Lecture ID'])

                    ## prompts, reflections, concepts, extractive summaries, abstractive summaries
                    ''' prompts '''
                    try:
                        prompts.append(data_lecture['Prompt'])
                    except Exception as e:

                        if 'INTER' in lecture or 'q3' in lecture or 'q8' in lecture:
                            prompts.append('Describe what you found most interesting in today’s class')
                        elif 'CONF' in lecture or 'q1' in lecture or 'q7' in lecture:
                            prompts.append('Describe what you found most confusing in today’s class')
                        else:
                            prompts.append('')
                            logging.error(f"no prompt for lecture {lecture} "+ str(e))

                    ''' Reflections '''
                    try:
                        reflections.append('\n'.join(
                            [' '.join(key.split(':')[-1].split()) for key, _ in data_lecture['Reflections'].items()]))
                    except Exception as e:
                        logging.error(f" no reflections for lecture {lecture} " + str(e))
                        reflections.append("")

                    ''' Abstractive Summary '''
                    try:
                        abstractive_summ.append(data_lecture['Abstractive Summary'])

                    except Exception as e:
                        logging.error(f" no abstractive summary for lecture {lecture} " + str(e))
                        abstractive_summ.append("")

                    ''' Extractive Summary '''
                    try:
                        _ixs = data_lecture['Extractive Summary']

                        _reflections = [_sent for _sent in [' '.join(key.split(':')[-1].split()) for key, _ in
                                                            data_lecture['Reflections'].items()]]
                        try:
                            # basic case with indexes
                            extractive_summ.append(
                                ' '.join([_reflections[int(i)] for i in data_lecture['Extractive Summary']]))
                        except:
                            # the other case where sentences present
                            logging.info(f" summary exist as sentences not indeces ...")
                            extractive_summ.append(' '.join(data_lecture['Extractive Summary']))

                    except Exception as e:
                        logging.error(f" no extractive summary for lecture {lecture} " + str(e))
                        extractive_summ.append("")

                    ''' Conceps  '''
                    try:
                        concepts.append(
                            list(set([concept for _, concept_dict in data_lecture['Reflections'].items() for _, concept in
                             concept_dict.items() if concept != ''])))

                    except Exception as e:
                        logging.error(f"error in adding conecpts for lecture {lecture}" + str(e))

                    ''' Phrase summary '''

                    try:
                        phrase_summ.append(
                            '\n'.join([phrase_dict['Phrase'] for phrase_dict in data_lecture['Phrase Summary']])
                        )

                    except Exception as e:
                        logging.error(f"error in geting phrase summaries for lecture {lecture} "+ str(e))




        except:
            logging.info(f"Course {course} has no annotated files")
            continue

print(len(reflections), len(courses), len(extractive_summ))

pd.DataFrame({'annotator': annotators,
              'course': courses,
              'lecture number': lecture_ids,
              'prompt': prompts,
              'reflections': reflections,
              'abstractive summary': abstractive_summ,
              'extractive summary': extractive_summ,
              'phrase summary': phrase_summ,
              'concepts':concepts}).to_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/summarization_table.csv', index=False)

