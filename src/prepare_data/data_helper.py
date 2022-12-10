def get_reflection(file):
    reflection = []
    for line in open(file, 'r').readlines():
        reflection.append(line.strip())
        if '@' in line:
            break

    return ' '.join(reflection)


def get_abs_summ(file):
    summary = []
    start_abs = False
    for line in open(file, 'r').readlines():
        if start_abs:
            if '@' not in line:
                summary.append(line.strip())
            elif '@' in line:
                break
        if '@highlight' in line:
            start_abs = True

    return ' '.join(summary)

def get_concept(file):
    concepts = []
    start_concept = False
    for line in open(file,'r').readlines():
        if start_concept:
           if '@' not in line:
               concepts.append(line.strip())
           else:
              break

        if '@concept' in line:
            start_concept = True

    return  list(set(concepts))