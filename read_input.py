import os

def load_train_data(root_directory):
    print(root_directory)
    classes = []
    reviews = [[] for i in range(4)]
    paths = []
    for (root,dirs,files) in os.walk(root_directory, topdown=True):
        if len(dirs)==0 and any(fname.endswith('.txt') for fname in files):
            c1 = root.split("\\")[-3]
            c2 = "."+ root.split("\\")[-2]
            if c1+c2 not in classes:
                classes.append(c1+c2)
            for file in files:
                with open(f"{root}\{file}",'r') as f:
                    paths.append(f"{root}\{file}")
                    lines = f.readlines()
                reviews[classes.index(c1+c2)].append(str(lines))
    return classes, reviews, paths

def load_test_data(root_directory):
    reviews = []
    paths = []
    for (root,dirs,files) in os.walk(root_directory, topdown=True):
        if len(dirs)==0 and any(fname.endswith('.txt') for fname in files):
            for file in files:
                with open(f"{root}\{file}",'r') as f:
                    paths.append(f"{root}\{file}")
                    lines = f.readlines()
                reviews.append(str(lines))
            
    return reviews, paths