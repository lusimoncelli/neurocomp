import os
import nibabel as nib
import numpy as np
import nibabel as nib
from nilearn import image, masking, plotting
from nilearn.decoding import SearchLight
from nilearn.input_data import NiftiMasker
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import torch
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

    # ajusto el modelo lineal
    training_subs = 100
    training_set = np.reshape(reduced[:training_subs], (reduced.shape[1]*training_subs, reduced.shape[-1]))
    ys = np.tile(aligned_embeddings.T, 10).T
    print('Test shape: ', training_set.shape)
    print('Expected shape: ', ys.shape)

    reg = LinearRegression()
    reg.fit(training_set, ys)

    thresh = 0.8
    for i, red in enumerate(reduced):
        prediction = reg.predict(red)
        similarity = cosine_similarity(prediction, aligned_embeddings)
        correct_predictions = np.sum(np.diag(similarity) >= thresh)
        total_predictions = similarity.shape[0]
        accuracy = correct_predictions / total_predictions * 100
        s = '' if i <= training_subs else 'not '
        print(f'Accuracy for sub {i} ({s}part of the training set): ', f'{accuracy:.2f}%')


    

    