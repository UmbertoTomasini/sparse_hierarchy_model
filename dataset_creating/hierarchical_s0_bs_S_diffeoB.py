import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import time
from itertools import *
import random
import numpy as np
from math import factorial
from utils_data import unique, dec2bin

def add_spec(tmp,spec,s,s0):
    for j in range(s):
        if j==0:    
            tmp = [spec]*s0 + tmp
        else:
            tmp = tmp[:(s0+1)*j] + [spec]*s0 + tmp[j*(s0+1):]
    return tmp

def hierarchical_features(num_features, num_layers, m, num_classes, s, s0,seed=0):
    """
    Build hierarchy of features.

    :param num_features: number of features to choose from at each layer (short: `n`).
    :param num_layers: number of layers in the hierarchy (short: `l`)
    :param m: features multiplicity (number of ways in which a feature can be made from sub-feat.)
    :param num_classes: number of different classes
    :param seed: sampling sub-features seed
    :return: features hierarchy as a list of length num_layers
    """
    random.seed(seed)
    features = [torch.arange(num_classes)]
    for l in range(num_layers):
        previous_features = features[-1].flatten()
        features_set = list(set([i.item() for i in previous_features]))
        num_layer_features = len(features_set)
        print(l)
        print(num_layer_features)
        # new_features = list(combinations(range(num_features), 2))
        list_elems = []
        for j in range(s):
            list_elems.append(list(range(num_features)))
            
        new_features = list(product(*list_elems))
        #new_features = list(product(range(num_features), range(num_features)))

        spec = num_features
        for i in range(len(new_features)):

            tmp = list(new_features[i])

            tmp = add_spec(tmp,spec,s,s0)
            tmp = tuple(tmp)
            new_features[i] = tmp


        if l>0:
            assert (
                len(new_features) >= m * (num_layer_features-1)
            )
        else:
            assert(
                len(new_features) >= m * (num_layer_features)
            ), "Not enough features to choose from!!"
        random.shuffle(new_features)
        #print(new_features.size())
        if s0>0:
            if l>0:
                new_features = new_features[: m * (num_layer_features-1)]
            else:
                new_features = new_features[: m * (num_layer_features)]
        else:
            new_features = new_features[: m * (num_layer_features)]
        new_features = list(sum(new_features, ()))  # tuples to list


        new_features = torch.tensor(new_features)
        new_features = new_features.reshape(-1, m, s*(s0+1))  # [n_features h-1, m, 4]
        if l>0 and s0>0:
            new_features = torch.cat((new_features, torch.ones((1,m,s*(s0+1)))*spec),0)

        # map features to indices
        feature_to_index = dict([(e, i) for i, e in enumerate(features_set)])
        indices = [feature_to_index[f.item()] for f in previous_features]

        new_features = new_features[indices]
        #print(new_features)
        features.append(new_features)
    #print(features)
    return features

def create_perms(s,s0):
    list_ps = []
    for k in range(s):
        ps2 = []
        list_order = [j for j in range(k*(s0+1),(k+1)*(s0+1))]
        for i in range(s0+1):
            list_order_new = list_order.copy()
            list_order_new [i] = list_order [s0]
            list_order_new [s0] = list_order [i]
            ps2.append(list_order_new)
        list_ps.append(ps2)

    ps0 = list(product(*list_ps))
    perms = []
    for a in range(len(ps0)):
        for q in range(s):
            if q==0:
                tmp = list(ps0[a][0])
            else:
                tmp = tmp+ list(ps0[a][q])
        perms.append(tmp)
    return perms


def generate_valid_permutations(s, s0):
    # Define the range of numbers
    upper_limit = s * (s0 + 1)
    numbers = list(range(1, upper_limit + 1))

    # Identify multiples of s0+1
    multiples = sorted([n for n in numbers if n % (s0 + 1) == 0])

    # Identify non-multiples
    non_multiples = [n for n in numbers if n % (s0 + 1) != 0]

    # Generate all permutations of non-multiples
    #non_multiples_perms = non_multiples #permutations(non_multiples)

    # Function to insert multiples into a permutation at valid positions
    def insert_multiples(perm, multiples, start=0):
        if not multiples:
            # Convert tuple to list and subtract 1 from each element
            return [list(map(lambda x: x - 1, perm))]

        inserted_perms = []
        for i in range(start, len(perm) + 1):
            new_perm = perm[:i] + [multiples[0],] + perm[i:]
            inserted_perms.extend(insert_multiples(new_perm, multiples[1:], i + 1))
        
        return inserted_perms

    # Generate all valid permutations as lists with adjustment
    valid_perms = []
    #for perm in non_multiples_perms:
    #    valid_perms.extend(insert_multiples(perm, multiples))
    valid_perms.extend(insert_multiples(non_multiples, multiples))

    return valid_perms





#def features_to_data(features, m, num_classes, num_layers, samples_per_class,s0, seed=0):
def features_to_data(features, m, num_classes, num_layers, samples_per_class, s, s0, bs=1, idx_bs=0, seed=21, seed_reset_layer_sem=42):
    """
    Build hierarchical dataset from features hierarchy.

    :param features: hierarchy of features
    :param m: features multiplicity (number of ways in which a feature can be made from sub-feat.)
    :param num_classes: number of different classes
    :param num_layers: number of layers in the hierarchy (short: `l`)
    :param samples_per_class: self-expl.
    :param seed: controls randomness in sampling
    :return: dataset {x, y}
    """
    start_time = time.time()
    #samples_per_class = 1
    np.random.seed(seed)
    #print(features[-1])
    x = features[-1].reshape(num_classes, *sum([(m, s*(s0+1)) for _ in range(num_layers)], ()))
    #print(x)
    y = torch.arange(num_classes)[None].repeat(samples_per_class, 1).t().flatten()

    perms = generate_valid_permutations(s, s0) #OK

    indices = []
    for l in range(num_layers):

        # randomly choose sub-features
        flag =  l >= seed_reset_layer_sem
        np.random.seed(seed + l  + flag + bs*idx_bs)

        random_features = np.random.choice(
            range(m), size=(samples_per_class * num_classes, (s*(s0+1)) ** l)
        ) .repeat((s*(s0+1)) ** (num_layers - l - 1), 1)
        indices.append(torch.tensor(random_features).long())
        
        range_choices = len(perms)#(s0+1)**s
        random_couples_single = np.random.choice(
            range(range_choices), size=(samples_per_class * num_classes, (s*(s0+1)) ** l)
        )#.repeat(4 ** (num_layers - l - 2), 1)

        if l != (num_layers-1):
            #print("first part")
            # indexing the left AND right sub-features
            # Repeat is there such that higher level features are chosen consistently for a give data-point

            l2 = l+1
            perm_layer = []
            perms_t = torch.tensor(perms)
            for (ia, a) in enumerate(random_couples_single):
                
                perms_t1 = perms_t[a].flatten().repeat((s*(s0+1)) ** (num_layers - l2 - 1), 1)
                perms_t1= (
                        perms_t1
                        .reshape((s*(s0+1)) ** (num_layers - l2 - 1), -1)
                        .t()
                        .flatten()
                    )
                perms_t1 = perms_t1.unsqueeze(0)
                perm_layer.append(perms_t1)
            perm_layer = torch.cat(perm_layer)
            indices.append(perm_layer)
            #print("layer: "+str(l))
            #print(time.time()-start_time)

        elif l==(num_layers-1):
            perm_last=[]
            for (idxa,a) in enumerate(random_couples_single):
                perm_last_j=[]
                for j in range((s*(s0+1))**l):

                    tmp_j = [(s*(s0+1))*j]*(s*(s0+1))
                    tmp_j_2 = [sum(value) for value in zip(perms[a[j]], tmp_j)]
                    perm_last_j.append(torch.tensor(tmp_j_2))
        
                perm_last .append(torch.cat(perm_last_j).unsqueeze(0))
            perm_last = torch.cat(perm_last)
            #print("layer: "+str(l))
            #print(time.time()-start_time)
            #print(torch.tensor(perm_last).size())
            #indices.append(torch.tensor(perm_layer))
        #indices.append(left_right)

    yi = y[:, None].repeat(1, (s*(s0+1)) ** (num_layers - 1))

    #x1 = x[tuple([yi.long(), *indices])]#.flatten(1)
    #print(*indices)
    #print(perm_last)
    #print(x.flatten(1))
    x = x[tuple([yi.long(), *indices])].flatten(1)
    #print(x)
    #print(x.size())
    for c in range(x.size(0)):
        x[c,:] = x[c,perm_last[c]]
    #x[:,:] = x[:, perm_last]
    #print(x)


    return x, y


class HierarchicalDataset(Dataset):
    """
    Hierarchical dataset.
    """

    def __init__(
        self,
        num_features=8,
        m=2,  # features multiplicity
        num_layers=2,
        num_classes=2,
        seed=0,
        train=True,
        input_format='onehot',
        whitening=0,
        transform=None,
        testsize=-1,
        memory_constraint=5e4,
        s=2,
        s0 = 0,
        bs =1,
        idx_bs = 0,
        seed_reset_layer_sem = 42,
        unique_datapoints = 1
    ):
        assert testsize or train, "testsize must be larger than zero when generating a test set!"
        torch.manual_seed(seed)
        self.num_features = num_features
        self.m = m  # features multiplicity
        self.num_layers = num_layers
        self.num_classes = num_classes
        #Pmax = (4*m) ** (2 ** num_layers - 1) * num_classes
        self.s0 = s0
        self.s = s
        #print(10*Pmax,memory_constraint)
        #samples_per_class = min(10 * Pmax, int(memory_constraint)) # constrain dataset size for memory budget
        samples_per_class = int(memory_constraint)
        features = hierarchical_features(
            num_features, num_layers, m, num_classes, s, s0, seed=seed
        )
        #self.x, self.targets = features_to_data(
        #    features, m, num_classes, num_layers, samples_per_class=samples_per_class,s0=s0, seed=seed
        #)
        self.x, self.targets = features_to_data(
            features, m, num_classes, num_layers, samples_per_class=samples_per_class,
            bs = bs , idx_bs = idx_bs, s =s ,s0=s0, seed=seed, seed_reset_layer_sem=seed_reset_layer_sem
        )
        #print(self.x)
        if unique_datapoints:
            
            self.x, unique_indices = unique(self.x, dim=0)
        
            self.targets = self.targets[unique_indices]
        #print(self.x)
        print(f"Data set size: {self.x.shape[0]}")

        # encode input pairs instead of features
        if "pairs" in input_format:
            self.x = pairing_features(self.x, num_features)

        #if 'onehot' not in input_format:
         #   assert not whitening, "Whitening only implemented for one-hot encoding"

        if "binary" in input_format:
            self.x = dec2bin(self.x)
            self.x = self.x.permute(0, 2, 1)
        elif "long" in input_format:
            self.x = self.x.long() + 1
        elif "decimal" in input_format:
            self.x = ((self.x[:, None] + 1) / num_features - 1) * 2
        elif "onehot" in input_format:
            self.x = F.one_hot(self.x.long()).float()
            self.x = self.x.permute(0, 2, 1)
            #self.x *= (1/(num_features))

            if whitening:
                inv_sqrt_n = (num_features +1 - 1) ** -.5
                self.x = self.x * (1 + inv_sqrt_n) - inv_sqrt_n
            else:
                exp = int("pairs" in input_format) + 1
                self.x *= num_features ** exp

        elif "all0" in input_format:
            #print('check2')
            self.x = F.one_hot(self.x.long()).float()
            self.x = self.x.permute(0, 2, 1)
            #print(self.x)
            if s0>0: 
                self.x = self.x[:,:-1,:]
            #print(self.x)
            #self.x *= (1/(num_features))

            if whitening:
                inv_sqrt_n = (num_features - 1) ** -.5
                self.x = self.x * (1 + inv_sqrt_n) - inv_sqrt_n
            else:
                exp = int("pairs" in input_format) + 1
                self.x *= num_features ** exp
        else:
            raise ValueError
        #print(self.x)
        if testsize == -1:
            testsize = min(len(self.x) // 5, 20000)

        P = torch.randperm(len(self.targets))
        if train and testsize:
            P = P[:-testsize]
        else:
            P = P[-testsize:]

        self.x, self.targets = self.x[P], self.targets[P]

        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return (torch.tensor, torch.tensor): (sample, label)
        """

        x, y = self.x[idx], self.targets[idx]

        if self.transform:
            x, y = self.transform(x, y)

        # if self.background_noise:
        #     g = torch.Generator()
        #     g.manual_seed(idx)
        #     x += torch.randn(x.shape, generator=g) * self.background_noise

        return x, y

def pairs_to_num(xi, n):
    """
        Convert one long input with n-features encoding to n^2 pairs encoding.
    """
    ln = len(xi)
    xin = torch.zeros(ln // 2)
    for ii, xii in enumerate(xi):
        xin[ii // 2] += xii * n ** (1 - ii % 2)
    return xin

def pairing_features(x, n):
    """
        Batch of inputs from n to n^2 encoding.
    """
    xn = torch.zeros(x.shape[0], x.shape[-1] // 2)
    for i, xi in enumerate(x.squeeze()):
        xn[i] = pairs_to_num(xi, n)
    return xn
