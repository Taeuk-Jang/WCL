from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


from random import sample 
import cv2
import numpy as np
import torch
import os
import pandas as pd


class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)

            return img, img2, target
        else:
            return img, target
    
class ImageDatasetPair(Dataset):
    def __init__(self, dataname, sens_idx, label_idx, mode = 'train', transform = None):
        self.dataname = dataname
        
        if dataname == 'celeba':
            self.root_dir = '/data/celebA/CelebA'
            self.data = pd.read_csv(os.path.join(self.root_dir, 'Anno/list_attr_celeba.csv'))
            self.split_data = pd.read_csv(os.path.join(self.root_dir, 'Anno/list_eval_partition.csv'))
            
            if mode == 'train':
                self.data = self.data[self.split_data['partition'] == 0].reset_index(drop=True)
            elif mode == 'valid':
                self.data = self.data[self.split_data['partition'] == 1].reset_index(drop=True)
            elif mode == 'test':
                self.data = self.data[self.split_data['partition'] == 2].reset_index(drop=True)
        
        
            self.root_dir = os.path.join(self.root_dir, 'Img/img_align_celeba')
            self.data['img_filename'] = self.data['image_id']
            
        elif dataname == 'CUB':
            self.root_dir = '/data/waterbird_complete95_forest2water2'
            self.metadata_df = pd.read_csv(os.path.join(self.root_dir, 'metadata.csv'))      
            
            if mode == 'train':
                self.data = self.metadata_df[self.metadata_df['split'] == 0].reset_index(drop=True)
            elif mode == 'valid':
                self.data = self.metadata_df[self.metadata_df['split'] == 1].reset_index(drop=True)
            elif mode == 'test':
                self.data = self.metadata_df[self.metadata_df['split'] == 2].reset_index(drop=True)
        
        self.transform = transform
        self.sens_idx = sens_idx
        self.label_idx = label_idx
        self.classes = self.data[self.label_idx].unique()
        self.sens_groups = self.data[self.sens_idx].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.open(os.path.join(self.root_dir, self.data['img_filename'][idx]))
        
        if isinstance(self.sens_idx, list):
            sens = np.array([self.data[s][idx] for s in self.sens_idx])
            sens = sum(sens * np.array([2 ** i for i in range(len(self.sens_idx))]))
        else:
            sens = int((self.data[self.sens_idx][idx] + 1) / 2)
            
        y = int((self.data[self.label_idx][idx] + 1) / 2)

        if self.transform:
            pos_1 = self.transform(image)
            pos_2 = self.transform(image)

        return  pos_1, pos_2, sens, y

class ImageDataset(Dataset):
    def __init__(self, dataname, sens_idx, label_idx, mode = 'train', transform = None):
        self.dataname = dataname
        
        if dataname == 'celeba':
            self.root_dir = '/data/celebA/CelebA'
            self.data = pd.read_csv(os.path.join(self.root_dir, 'Anno/list_attr_celeba.csv'))
            self.split_data = pd.read_csv(os.path.join(self.root_dir, 'Anno/list_eval_partition.csv'))
            
            if mode == 'train':
                self.data = self.data[self.split_data['partition'] == 0].reset_index(drop=True)
            elif mode == 'valid':
                self.data = self.data[self.split_data['partition'] == 1].reset_index(drop=True)
            elif mode == 'test':
                self.data = self.data[self.split_data['partition'] == 2].reset_index(drop=True)
        
            self.root_dir = os.path.join(self.root_dir, 'Img/img_align_celeba')
            self.data['img_filename'] = self.data['image_id']
            
        elif dataname == 'CUB':
            self.root_dir = '/data/waterbird_complete95_forest2water2'
            self.metadata_df = pd.read_csv(os.path.join(self.root_dir, 'metadata.csv'))      
            
            if mode == 'train':
                self.data = self.metadata_df[self.metadata_df['split'] == 0].reset_index(drop=True)
            elif mode == 'valid':
                self.data = self.metadata_df[self.metadata_df['split'] == 1].reset_index(drop=True)
            elif mode == 'test':
                self.data = self.metadata_df[self.metadata_df['split'] == 2].reset_index(drop=True)
        
        self.transform = transform
        self.sens_idx = sens_idx
        self.label_idx = label_idx
        self.classes = self.data[self.label_idx].unique()
        self.sens_groups = self.data[self.sens_idx].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.open(os.path.join(self.root_dir, self.data['img_filename'][idx]))
        
        if isinstance(self.sens_idx, list):
            sens = np.array([self.data[s][idx] for s in self.sens_idx])
            sens = sum(sens * np.array([2 ** i for i in range(len(self.sens_idx))]))
        else:
            sens = int((self.data[self.sens_idx][idx] + 1) / 2)
            
        y = int((self.data[self.label_idx][idx] + 1) / 2)

        if self.transform:
            image = self.transform(image)

        return  image, sens, y
    
class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class CIFAR100Pair_true_label(CIFAR100):
    #dataloader where pairs of positive samples are randomly sampled from pairs
    #of inputs with the same label. 
    def __init__(self, root='../data', train=True, transform=None):
        super().__init__(root=root, train=train, transform=transform)
        def get_labels(i):
            return [index for index in range(len(self)) if self.targets[index]==i]

        self.label_index = [get_labels(i) for i in range(100)]

    def __getitem__(self, index):
        img1, target = self.data[index], self.targets[index]

        index_example_same_label=sample(self.label_index[self.targets[index]],1)[0]
        img2 = self.data[index_example_same_label]

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform is not None:
            pos_1 = self.transform(img1)
            pos_2 = self.transform(img2)
 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR100Pair(CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def get_transform(dataset_name):
    if dataset_name in ['cifar100', 'cifar10', 'stl10']:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    elif dataset_name == 'celeba':
        img_size = 224
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif dataset_name == 'cub':
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        assert target_resolution is not None

        # Resizes the image to a slightly larger square then crops the center.
        test_transform = transforms.Compose([
            transforms.Resize((
                int(target_resolution[0] * scale),
                int(target_resolution[1] * scale),
            )),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    elif dataset_name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
                
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif dataset_name == 'tiny-imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
    return transform, test_transform


class tabular_dataset(Dataset):
    def __init__(self, X, Y, A, std = None, noise = True, M=2):
        self.X_raw = X
        self.Y = Y
        self.A = A
        self.noise = noise
        self.std = std
        self.M = M
        
#         if noise:
#             self.X = []
#             self.X.append((self.X_raw + np.random.normal(0,std,self.X_raw.shape), 0, 1))
#             self.X.append((self.X_raw + np.random.normal(0,std,self.X_raw.shape), 0, 1))
#         else:
#             self.X = self.X_raw
            
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        
        if self.noise:
            return [self.X_raw[idx] + np.random.normal(0, self.std) for i in range(self.M)], self.Y[idx], self.A[idx]
        else:
            return self.X_raw[idx], self.Y[idx], self.A[idx]

# def get_dataset(name, save=False, corr_sens=False, seed=42, verbose=False):
#     """
#     Retrieve dataset and all relevant information
#     :param name: name of the dataset
#     :param save: if set to True, save the dataset as a pickle file. Defaults to False
#     :return: Preprocessed dataset and relevant information
#     """
#     def get_numpy(df):
#         new_df = df.copy()
#         cat_columns = new_df.select_dtypes(['category']).columns
#         new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
#         return new_df.values

#     if name == 'adult':
#         # Load data
#         feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', \
#                          'marital-status', 'occupation', 'relationship', 'race', 'sex', \
#                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
#         df = pd.read_csv('/data/adult/adult.data', names=feature_names)
        
#         if verbose:
#             print('Raw Dataset loaded.')
#         num_train = df.shape[0]
#         pos_class_label = ' >50K'
#         neg_class_label = ' <=50K'
#         y = np.zeros(num_train)
#         y[df.iloc[:,-1].values == pos_class_label] = 1
# #         df = df.drop(['fnlwgt', 'education-num'], axis=1)
#         df = df.drop(['fnlwgt'], axis=1)
#         num_var_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
# #         cat_var_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
#         cat_var_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
#         feature_names = num_var_names + cat_var_names
#         df = df[feature_names]
#         df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship'], prefix_sep='=')
#         if verbose:
#             print('Selecting relevant features complete.')

#         for col in df:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].astype('category')
#             else:
#                 df[col] = df[col].astype(float)

#         dtypes = df.dtypes

#         X = get_numpy(df)
#         if verbose:
#             print('Numpy conversion complete.')

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
#         if verbose:
#             print('Dataset split complete.')

#         # sens idx
#         race_idx = df.columns.get_loc('race')
#         sex_idx = df.columns.get_loc('sex')
# #         print( df.columns.get_loc('sex'))
#         sens_idc = [race_idx, sex_idx]

#         race_cats = df.iloc[:, race_idx].cat.categories
#         sex_cats = df.iloc[:, sex_idx].cat.categories
        
#         if verbose:
#             print(race_cats, sex_cats)

#         # Remove sensitive information from data
#         X_train_removed = np.delete(X_train, sens_idc , 1)
#         X_test_removed = np.delete(X_test, sens_idc , 1)
#         dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

#         if verbose:
#             print('Senstive attribute removal complete.')

#     elif name == 'bank':
#         # Load data
        
#         df = pd.read_csv('/data/bank/bank-additional-full.csv', sep = ';', na_values=['unknown'])

#         df['age'] = df['age'].apply(lambda x: x >= 25)
#         df = df[np.array(df.default == 'no') + np.array(df.default == 'yes')]

#         #         num_var_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
#         cat_var_names = ['job', 'marital', 'education', 'default',
#                          'housing', 'loan', 'contact', 'month', 'day_of_week',
#                      'poutcome']
#         #         feature_names = num_var_names + cat_var_names
#         #         df = df[feature_names]

# #         df = df.drop(['default'], axis=1)
#         df = pd.get_dummies(df, columns=cat_var_names, prefix_sep='=')



#         for col in df:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].astype('category')
#             else:
#                 df[col] = df[col].astype(float)


#         Xy = get_numpy(df)

#         idx = np.zeros(Xy.shape[-1]).astype(bool)
#         idx[df.columns.get_loc('y')] = 1

#         X = Xy[:, ~idx]
#         y = Xy[:, idx].reshape(-1)       

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        
#         dtypes = df.dtypes[~idx]

#         # sens idx
#         sex_idx = df.columns.get_loc('age')
#         race_idx = df.columns.get_loc('age')
#         sens_idc = [sex_idx]


#         # Remove sensitive information from data
#         X_train_removed = np.delete(X_train, sens_idc , 1)
#         X_test_removed = np.delete(X_test, sens_idc , 1)
#         dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

#         if verbose:
#             print('Senstive attribute removal complete.')
            
#     elif name == 'compas':
        
#         def quantizeLOS(x):
#             if x<= 7:
#                 return '<week'
#             if 8<x<=93:
#                 return '<3months'
#             else:
#                 return '>3 months'


#                 # Load data
#         df = pd.read_csv('/data/compas/compas-scores-two-years.csv', index_col='id', na_values=[])

#         df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'juv_fel_count',
#                   'juv_misd_count', 'juv_other_count', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score','c_charge_desc',
#                  'two_year_recid', 'c_jail_in', 'c_jail_out']]

#         # Indices of data samples to keep
#         ix = df['days_b_screening_arrest'] <= 30
#         ix = (df['days_b_screening_arrest'] >= -30) & ix
#         # ix = (df['is_recid'] != -1) & ix
#         ix = (df['c_charge_degree'] != "O") & ix
#         ix = (df['score_text'] != 'N/A') & ix
#         df = df.loc[ix,:]
#         df['length_of_stay'] = abs(pd.to_datetime(df['c_jail_out'])-
#                                 pd.to_datetime(df['c_jail_in'])).apply(
#                                                         lambda x: x.days)

#         # Restrict races to African-American and Caucasian
#         dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

#         # Restrict the features to use
#         dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count',
#                 'length_of_stay', 'two_year_recid','c_charge_desc']].copy()
#         dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))

#         num_train = dfcutQ.shape[0]

#         num_var_names = ['two_year_recid', 'sex','race', 'score_text','priors_count', 'length_of_stay','c_charge_desc' ]
#         categorical_features = ['age_cat','c_charge_degree']

#         dfcutQ = pd.get_dummies(dfcutQ, columns=categorical_features, prefix_sep='=')

#         for col in dfcutQ:
#             if dfcutQ[col].dtype == 'object':
#                 dfcutQ[col] = dfcutQ[col].astype('category')
#             else:
#                 dfcutQ[col] = dfcutQ[col].astype(float)


#         pos_class_label = 1
#         neg_class_label = 0

#         idx = np.zeros(dfcutQ.shape[1]).astype(bool)
#         y_idx = dfcutQ.columns.get_loc('two_year_recid')
#         idx[y_idx] = True

#         Xy = get_numpy(dfcutQ)

#         X = Xy[:, ~idx]
#         y = Xy[:, idx].reshape(-1)

#         #remove bad quality sample
#         idx = X[:, 5] == -1

#         X = X[~idx, :]
#         y = y[~idx]

#         dfcutQ = dfcutQ.drop(['two_year_recid'], axis = 1)
        
#         dtypes = dfcutQ.dtypes

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

#         if verbose:
#             print('Dataset split complete.')
            
#         race_idx = dfcutQ.columns.get_loc('race')
#         sex_idx = dfcutQ.columns.get_loc('sex')
        
#         sens_idc = [race_idx, sex_idx]
        
#         race_cats = dfcutQ.iloc[:,race_idx].cat.categories
#         sex_cats = dfcutQ.iloc[:,sex_idx].cat.categories
        
#         if verbose:
#             print(race_cats, sex_cats)

#         X_train_removed = np.delete(X_train, sens_idc , 1)
#         X_test_removed = np.delete(X_test, sens_idc , 1)
#         dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

#         if verbose:
#             print('Senstive attribute removal complete.')
    
#     elif name == 'meps':
        
#         df = pd.read_csv('/data/meps/h181.csv', sep=',', na_values=[])
        
#         def race(row):
#             if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
#                 return 'White'
#             return 'Non-White'
        
#         def sex(row):
#             if row['SEX'] == 1:
#                 return 'female'
#             return 'male'

#         df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
#         df['SEX'] = df.apply(lambda row: sex(row), axis=1)
        
#         df = df.rename(columns = {'RACEV2X' : 'RACE'})

#         df = df[df['PANEL'] == 19]

#         df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
#                               'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
#                               'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
#                               'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
#                               'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

#         df = df[df['REGION'] >= 0] # remove values -1
#         df = df[df['AGE'] >= 0] # remove values -1

#         df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

#         df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

#         df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
#                                  'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
#                                  'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
#                                  'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
#                                  'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1
                
#         def utilization(row):
#             return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

#         df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
#         lessE = df['TOTEXP15'] < 10.0
#         df.loc[lessE,'TOTEXP15'] = 0.0
#         moreE = df['TOTEXP15'] >= 10.0
#         df.loc[moreE,'TOTEXP15'] = 1.0

#         df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
        
#         features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
#                                  'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
#                                  'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
#                                  'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
#                                  'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42','PCS42',
#                                  'MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION']
        
#         categorical_features=['REGION','SEX', 'MARRY',
#              'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
#              'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
#              'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
#              'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
#              'PHQ242','EMPST','POVCAT','INSCOV']
        
#         df = df[features_to_keep]
#         df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')
        
#         num_train = df.shape[0]

#         pos_class_label = 1
#         neg_class_label = 0
#         y = np.zeros(num_train)
        
#         verbose = True
        
#          # sens idx
#         race_idx = df.columns.get_loc('RACE')
#         sex_idx = df.columns.get_loc('RACE')
        
#         sens_idc = [race_idx]
         
#         for col in df:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].astype('category')
#             else:
#                 df[col] = df[col].astype(float)
                
        
        
#         idx = np.zeros(df.shape[1]).astype(bool)
#         y_idx = df.columns.get_loc('UTILIZATION')
#         idx[y_idx] = True
        
# #         min_max_scaler = MaxAbsScaler()
#         Xy = get_numpy(df)
#         X = Xy[:, ~idx]
#         y = Xy[:, idx].reshape(-1)
        
#         df = df.drop(['UTILIZATION'], axis=1)
#         dtypes = df.dtypes
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
#         if verbose:
#             print('Dataset split complete.')

#         # Remove sensitive information from data
        
#         race_cats = df['RACE'].cat.categories
# #         sex_cats = df[feature_names[sex_idx]].cat.categories
#         if verbose:
#             print(race_cats)
            
#         X_train_removed = np.delete(X_train, sens_idc , 1)
#         X_test_removed = np.delete(X_test, sens_idc , 1)
#         dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]


#         if verbose:
#             print('Senstive attribute removal complete.')

    
#     elif name == 'german':
#         # Download data if needed
#         _german_loan_attribute_map = dict(
#             A11='< 0 DM',
#             A12='0-200 DM',
#             A13='>= 200 DM',
#             A14='no checking',
#             A30='no credits',
#             A31='all credits paid back',
#             A32='existing credits paid back',
#             A33='delayed past payments',
#             A34='critical account',
#             A40='car (new)',
#             A41='car (used)',
#             A42='furniture/equipment',
#             A43='radio/television',
#             A44='domestic appliances',
#             A45='repairs',
#             A46='education',
#             A47='(vacation?)',
#             A48='retraining',
#             A49='business',
#             A410='others',
#             A61='< 100 DM',
#             A62='100-500 DM',
#             A63='500-1000 DM',
#             A64='>= 1000 DM',
#             A65='unknown/no sav acct',
#             A71='unemployed',
#             A72='< 1 year',
#             A73='1-4 years',
#             A74='4-7 years',
#             A75='>= 7 years',
#             #A91='male & divorced',
#             #A92='female & divorced/married',
#             #A93='male & single',
#             #A94='male & married',
#             #A95='female & single',
#             A91='male',
#             A92='female',
#             A93='male',
#             A94='male',
#             A95='female',
#             A101='none',
#             A102='co-applicant',
#             A103='guarantor',
#             A121='real estate',
#             A122='life insurance',
#             A123='car or other',
#             A124='unknown/no property',
#             A141='bank',
#             A142='stores',
#             A143='none',
#             A151='rent',
#             A152='own',
#             A153='for free',
#             A171='unskilled & non-resident',
#             A172='unskilled & resident',
#             A173='skilled employee',
#             A174='management/self-employed',
#             A191='no telephone',
#             A192='has telephone',
#             A201='foreigner',
#             A202='non-foreigner',
#         )

#         filename = '/data/german/german.data'
#         if not os.path.isfile(filename):
#             print('Downloading data to %s' % os.path.abspath(filename))
#             urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
#                                        filename)

#         # Load data and setup dtypes
#         col_names = [
#             'checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
#             'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
#             'other_debtors', 'residing_since', 'property', 'age',
#             'inst_plans', 'housing', 'num_credits',
#             'job', 'dependents', 'telephone', 'foreign_worker', 'status']
        
# #         AIF360
#         column_names = ['status', 'month', 'credit_history',
#             'purpose', 'credit_amount', 'savings', 'employment',
#             'investment_as_income_percentage', 'personal_status',
#             'other_debtors', 'residence_since', 'property', 'age',
#             'installment_plans', 'housing', 'number_of_credits',
#             'skill_level', 'people_liable_for', 'telephone',
#             'foreign_worker', 'credit']
        
#         df = pd.read_csv(filename, delimiter=' ', header=None, names=column_names)
        
#         def group_credit_hist(x):
#             if x in ['A30', 'A31', 'A32']:
#                 return 'None/Paid'
#             elif x == 'A33':
#                 return 'Delay'
#             elif x == 'A34':
#                 return 'Other'
#             else:
#                 return 'NA'

#         def group_employ(x):
#             if x == 'A71':
#                 return 'Unemployed'
#             elif x in ['A72', 'A73']:
#                 return '1-4 years'
#             elif x in ['A74', 'A75']:
#                 return '4+ years'
#             else:
#                 return 'NA'

#         def group_savings(x):
#             if x in ['A61', 'A62']:
#                 return '<500'
#             elif x in ['A63', 'A64']:
#                 return '500+'
#             elif x == 'A65':
#                 return 'Unknown/None'
#             else:
#                 return 'NA'
#         def group_status(x):
#             if x in ['A11', 'A12']:
#                 return '<200'
#             elif x in ['A13']:
#                 return '200+'
#             elif x == 'A14':
#                 return 'None'
#             else:
#                 return 'NA'

#         def group_purpose(x):
#             if x in ['A40', 'A41', 'A42', 'A43', 'A47', 'A410']:
#                 return 'non-essential'
#             elif x in ['A44', 'A45', 'A46', 'A48', 'A49']:
#                 return 'essential'

#         status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
#                       'A92': 'female', 'A95': 'female'}
#         df['sex'] = df['personal_status'].replace(status_map)

#         # group credit history, savings, and employment
#         df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
#         df['savings'] = df['savings'].apply(lambda x: group_savings(x))
#         df['employment'] = df['employment'].apply(lambda x: group_employ(x))
#     #     df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
#         df['purpose'] = df['purpose'].apply(lambda x:group_purpose(x))
#         df['status'] = df['status'].apply(lambda x: group_status(x))
        
#         cat_features = ['credit_history', 'savings', 'employment',  'purpose', 'other_debtors', 'property', 'housing', 'skill_level', \
#                 'investment_as_income_percentage', 'status', 'installment_plans', 'foreign_worker']
        
#         df = pd.get_dummies(df, columns=cat_features, prefix_sep='=')
#         df = df.drop(['telephone', 'personal_status',], axis = 1)
    
            
#         for col in df:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].astype('category')
#             else:
#                 df[col] = df[col].astype(float)
                
#         df['age'] = df['age'].apply(lambda x: x >= 25).astype('category')

#         def get_numpy(df):
#             new_df = df.copy()
#             cat_columns = new_df.select_dtypes(['category']).columns
#             new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
#             return new_df.values
        
#         y_idx = df.columns.get_loc('credit')
#         idx = np.zeros(df.shape[1]).astype(bool)
#         idx[y_idx] = True
        
#         Xy = get_numpy(df)
#         X = Xy[:,~idx]
#         y = Xy[:,idx].reshape(-1)
        
#         # Make 1 (good customer) and 0 (bad customer)
#         # (Originally 2 is bad customer and 1 is good customer)
#         sel_bad = y == 2
#         y[sel_bad] = 0
#         y[~sel_bad] = 1
#         feature_labels = df.columns.values[:-1]  # Last is prediction
#         dtypes = df.dtypes[~idx]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

#         # Senstivie attribute
# #         foreign = 19
#         age_idx = df.columns.get_loc('age')
#         sex_idx = df.columns.get_loc('sex')
#         sens_idc = [sex_idx, age_idx]
        
        
#         age_cats = df.iloc[:, age_idx].cat.categories
#         sex_cats = df.iloc[:, sex_idx].cat.categories
#         print([age_cats, sex_cats])

#         # Remove sensitive information from data
#         X_train_removed = np.delete(X_train, sens_idc , 1)
#         X_test_removed = np.delete(X_test, sens_idc , 1)
#         dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

#         race_idx = age_idx

#     else:
#         raise ValueError('Data name invalid.')

#     return X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, dtypes, dtypes_, sens_idc, race_idx, sex_idx

                    
def get_dataset(dataset_name, root='../data', pair=True):
    transform, test_transform = get_transform(dataset_name)
    if pair:
        if dataset_name=='cifar10':
            train_data = CIFAR10Pair(root=root, train=True, transform=transform, download = True)
            memory_data = CIFAR10Pair(root=root, train=True, transform=test_transform)
            test_data = CIFAR10Pair(root=root, train=False, transform=test_transform)
        elif dataset_name=='cifar100':
            train_data = CIFAR100Pair(root=root, train=True, transform=transform, download = True)
            memory_data = CIFAR100Pair(root=root, train=True, transform=test_transform)
            test_data = CIFAR100Pair(root=root, train=False, transform=test_transform)
        elif dataset_name=='stl10':
            train_data = STL10Pair(root=root, split='train+unlabeled', transform=transform, download = True)
            memory_data = STL10Pair(root=root, split='train', transform=test_transform)
            test_data = STL10Pair(root=root, split='test', transform=test_transform)
#         (self, dataname, sens_idx, label_idx, mode = 'train', transform = None):
        elif dataset_name=='celeba':
            train_data = ImageDatasetPair('celeba', 'Male', 'Attractive', 'train', transform)
            memory_data = ImageDatasetPair('celeba', 'Male', 'Attractive', 'train', test_transform)
            test_data = ImageDatasetPair('celeba', 'Male', 'Attractive', 'valid', test_transform)
        elif dataset_name=='cub':
            train_data = ImageDatasetPair('CUB', 'place', 'y', 'train', transform)
            memory_data = ImageDatasetPair('CUB', 'place', 'y', 'train', test_transform)
            test_data = ImageDatasetPair('CUB', 'place', 'y', 'valid', test_transform)
            
        elif dataset_name=='imagenet':
            train_dir = '/data/imagenet-100/train'
            val_dir = '/data/imagenet-100/val'
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            
            train_data = ImageFolderInstance(train_dir, transform=transform, two_crop = True)
            memory_data = ImageFolderInstance(train_dir, transform=transform, two_crop = True)
            test_data = ImageFolderInstance(val_dir, transform=test_transform, two_crop = True)
        
        elif dataset_name=='tiny-imagenet':
            train_dir = '/data/tiny-imagenet-200/train'
            val_dir = '/data/tiny-imagenet-200/val'
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            
            train_data = ImageFolderInstance(train_dir, transform=transform, two_crop = True)
            memory_data = ImageFolderInstance(train_dir, transform=transform, two_crop = True)
            test_data = ImageFolderInstance(val_dir, transform=test_transform, two_crop = True)
            
        elif dataset_name=='cifar100_true_label':
            train_data = CIFAR100Pair_true_label(root=root, train=True, transform=transform)
            memory_data = CIFAR100Pair_true_label(root=root, train=True, transform=test_transform)
            test_data = CIFAR100Pair_true_label(root=root, train=False, transform=test_transform)
        else:
            raise Exception('Invalid dataset name')
    else:
        if dataset_name in ['cifar10', 'cifar10_true_label']:
            train_data = CIFAR10(root=root, train=True, transform=transform)
            memory_data = CIFAR10(root=root, train=True, transform=test_transform)
            test_data = CIFAR10(root=root, train=False, transform=test_transform)
        elif dataset_name in ['cifar100', 'cifar100_true_label']:
            train_data = CIFAR100(root=root, train=True, transform=transform)
            memory_data = CIFAR100(root=root, train=True, transform=test_transform)
            test_data = CIFAR100(root=root, train=False, transform=test_transform)
        elif dataset_name=='stl10':
            train_data = STL10(root=root, split='train', transform=transform)
            memory_data = STL10(root=root, split='train', transform=test_transform)
            test_data = STL10(root=root, split='test', transform=test_transform)
        elif dataset_name=='celeba':
            train_data = ImageDataset('celeba', 'Male', 'Attractive', 'train', transform)
            memory_data = ImageDataset('celeba', 'Male', 'Attractive', 'train', test_transform)
            test_data = ImageDataset('celeba', 'Male', 'Attractive', 'valid', test_transform)
        elif dataset_name=='cub':
            train_data = ImageDataset('CUB', 'place', 'y', 'train', transform)
            memory_data = ImageDataset('CUB', 'place', 'y', 'train', test_transform)
            test_data = ImageDataset('CUB', 'place', 'y', 'valid', test_transform)
        elif dataset_name=='imagenet':
            train_dir = '/data/imagenet-100/train'
            val_dir = '/data/imagenet-100/val'
            
            train_data = ImageFolderInstance(train_dir, transform=transform , two_crop = False)
            memory_data = ImageFolderInstance(train_dir, transform= transform, two_crop = False)
            test_data = ImageFolderInstance(val_dir, transform=test_transform , two_crop = False)
        else:
            raise Exception('Invalid dataset name')

    return train_data, memory_data, test_data
        
