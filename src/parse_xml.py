import xml.etree.ElementTree as ET

def parse_xml():
    tree = ET.parse('../FewSumm/artifacts/ABSA_Gold_TestData/Restaurants_Test_Gold.xml')
    # tree = ET.parse('./ABSA_Gold_TestData/Laptops_Test_Gold.xml')
    root = tree.getroot()

    aspect_sentiment_pairs = []
    reviews = []
    for sentence in root:
        for child in sentence:
            if child.tag == 'aspectTerms':
                for aspect_term in child:
                    aspect = aspect_term.attrib['term']
                    sentiment = aspect_term.attrib['polarity']
                    aspect_sentiment_pairs.append((aspect, sentiment))
            if child.tag == 'text':
                reviews.append(child.text)
    
    # filter out all conflict labels
    i = 0
    num_conflict_labels = 0
    while i < len(aspect_sentiment_pairs):
        if aspect_sentiment_pairs[i][1] == 'conflict':
            aspect_sentiment_pairs.pop(i)
            num_conflict_labels += 1
        else:
            i += 1
    print('num conflict labels filtered out: ', num_conflict_labels)
    return reviews, aspect_sentiment_pairs

# parse_xml()
                



