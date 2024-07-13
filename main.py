import os
import nibabel as nib
import numpy as np
import nibabel as nib
from nilearn import image, masking, plotting
from nilearn.decoding import SearchLight
from nilearn.input_data import NiftiMasker
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import xgboost as xgb
import spacy
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

reduced_dir ='./reduced/'
sub_count = 20

def load_sub(sub_no):
    reduced_path = reduced_dir + f'sub-{sub_no:>02}.npy'

    print(f'Loading subject {sub_no}')

    if os.path.exists(reduced_path):
        return np.load(reduced_path)

    data_dir = './study-forrest/ds000113'
    mask_dir = './3mm_mask.nii.gz'

    fnames_data = [ f'{data_dir}/sub-{sub_no:>02}/ses-movie/func/sub-{sub_no:>02}_ses-movie_task-movie_run-{i}_bold.nii.gz' for i in range(1,9)]

    fmri_imgs = []
    fmri_data = []

    for f in fnames_data:
        img = nib.load(f)
        data = img.get_fdata()
        
        fmri_imgs.append(img)
        fmri_data.append(data)
        
    mask_img = nib.load(mask_dir)
    mask_data = mask_img.get_fdata()
    affine = nib.load(fnames_data[0]).affine    
    data = np.concatenate((fmri_data), axis = 3)
    img = nib.Nifti1Image(data, affine) # Create a new NiBabel image
    pca = PCA(n_components = 10) 
    masker = masking.compute_epi_mask(img)
    masked_data = masking.apply_mask(img, masker)
    reduced = pca.fit_transform(masked_data)
    np.save(reduced_path, reduced)
    return reduced


def load_all():
    for i in range(0, 20):
        print(f'Loading sub {i}')
        try:
            load_sub(i)
            print(f'Loaded sub {i}')
        except:
            print(f'Failed loading sub {i}')

def accuracy(actual, predicted):
    thresh = 0.75

    similarity = cosine_similarity(predicted, actual)
    correct_predictions = np.sum(np.diag(similarity) >= thresh)
    total_predictions = similarity.shape[0]
    return correct_predictions / total_predictions * 100

def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}%',
            ha='center',
            va='bottom'
        )

def closest(aligned_embeddings, predicted_embedding):
    idx, min = -1, np.inf
    for i, e in enumerate(aligned_embeddings):
        dist = np.linalg.norm(predicted_embedding - e)
        if dist < min:
            idx = i
            min = dist
    return idx

if __name__ == '__main__':
    print('*'*10)
    if not os.path.exists(reduced_dir):
        os.mkdir(reduced_dir)

    # csv with annotations
    print('Loading annotations')
    dialogue_file = "./study-forrest/ds000113/stimuli/annotations/german_audio_description.csv"
    text_data = pd.read_csv(dialogue_file, header=None)
    text_data.columns = ['start', 'end', 'text']
    texts = text_data['text']
    text_data.head()
    # nlp model deutsch
    nlp = spacy.load('de_core_news_sm')
    embeddings = []

    print('Calculating embeddings')
    for _, row in tqdm(text_data.iterrows(), total = text_data.shape[0]):
        doc = nlp(row['text'])
        embeddings.append(doc.vector)
    
    embeddings = np.array(embeddings)

    print('Loading subjects')
    reduced = []
    for i in range(0, sub_count):
        try:
            reduced.append(load_sub(i))
        except:
            pass # mi espiritu informático llora pero hay algunos sujetos que no están    
    
    print('Number of subjects: ', len(reduced))
    reduced = np.array(reduced)
    print('Reduced shape: ', reduced.shape)

    # TR = 2s (tiempo de adquisicion) y 3599 muestras temporales.
    TS = 2

    aligned_embeddings = []
    row = 0
    for frame in range(reduced.shape[1]):
        if text_data.iloc[row].end < TS*frame and row < text_data.shape[0] - 1:
            row += 1
        aligned_embeddings.append(embeddings[row])

    aligned_embeddings = np.array(aligned_embeddings)

    training_subs_sets = [10]
    for training_subs in training_subs_sets:
        np.random.shuffle(reduced)
        print(f'Test subjects: {training_subs}')
        training_set = np.reshape(reduced[:training_subs], (reduced.shape[1]*training_subs, reduced.shape[-1]))
        ys = np.tile(aligned_embeddings.T, training_subs).T
        print('Test shape: ', training_set.shape)
        print('Expected shape: ', ys.shape)

        models = [
            LinearRegression(), 
            # RandomForestRegressor(n_estimators=100, random_state=21), 
            # xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
            ]
        model_names = ['lin', 'rf', 'xgb']
        results = {}

        for name, model in zip(model_names, models):
            model.fit(training_set, ys)

            train_acc, test_acc = [], []
            for i, red in enumerate(reduced):
                predicted = model.predict(red)
                acc = accuracy(aligned_embeddings, predicted)
                train_acc.append(acc) if i < training_subs else test_acc.append(acc)        
                        
            results[name] = np.mean(train_acc), np.mean(test_acc)

        labels = list(results.keys())
        values = np.array(list(results.values()))

        # Number of entries
        n = len(labels)

        # Create an array for the x-axis locations of the groups
        x = np.arange(n)

        # Width of the bars
        width = 0.35

        # Create the bar graph
        fig, ax = plt.subplots()
        bar1 = ax.bar(x - width/2, values[:, 0], width, label='Training Set')
        bar2 = ax.bar(x + width/2, values[:, 1], width, label='Test Set')

        # Add labels, title, and legend
        ax.set_xlabel('Modelo')
        ax.set_ylabel('Precisión (%)')
        ax.set_title(f'Precisión por modelo (training set de {training_subs})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        add_value_labels(bar1)
        add_value_labels(bar2)

        # Show the plot
        plt.savefig(f'./res_{training_subs:>02}.png')
    