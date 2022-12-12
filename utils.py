import pickle
from itertools import product

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from Levenshtein import distance as lev_dist
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix


def load_df_geo():
    with open('dict_geo.pickle', 'rb') as handle:
        dict_geo = pickle.load(handle)

    df_geo = pd.DataFrame(dict_geo).T
    df_geo.columns = ['city', 'country']

    return df_geo


def calculate_inter_attack_times(df, groupby='src'):
    def f(group):
        group = group.sort_values('time')
        t = group.time.values.astype('datetime64[s]')
        dt = np.zeros(len(group))
        dt[1:] = pd.Series(t[1:] - t[:-1]).dt.seconds
        dt[0] = None
        return pd.Series(dt, index=group.index)

    grouped = df.groupby(groupby).apply(f)
    dt = grouped.reset_index(level=groupby)[0].rename('dt')

    return dt



def calculate_attack_count_cmf(df):
    counts = df.groupby('src').size().sort_values(ascending=False)
    cmf = np.cumsum(counts) / counts.sum()
    return list(counts.keys()), cmf



def plot_dt_density(df, src, t_max=np.inf, fit=False, alpha=1, label='hist'):
    dt_orig = df[df.src == src].dt
    t_max_percentile = ss.percentileofscore(dt_orig, t_max)
    t_max_percentile = round(t_max_percentile, 2)

    dt = dt_orig.dropna()
    dt = dt[dt <= t_max]
    plt.hist(dt, density=True, label=label, alpha=alpha)

    if fit:
        P = ss.expon.fit(dt)
        t = np.linspace(0, t_max)
        y = ss.expon.pdf(t, *P)
        plt.plot(t, y, label='exp fit')

        p = round(P[1], 3)

        return t_max_percentile, p
    else:
        return t_max_percentile



def plot_bot(df, bot, t_max=np.inf, fit=False, alpha=.5):
    rets = [plot_dt_density(df, ip, t_max, fit, alpha) for ip in bot]
    return rets



def print_text_features(df, src, top_n=5):
    df_src = df[df.src == src]
    print('IP:', src)
    print(f'origin: {df_src.iloc[0].city}, {df_src.iloc[0].country}')
    print('client_versions:', df_src.client_version.value_counts().index.to_list()[:top_n])
    print('usernames:', df_src.user.value_counts().index.to_list()[:top_n])
    print('passwords:', df_src.password.value_counts().index.to_list()[:top_n])



def d_to_set(x, Y, d):
    return min(d(x, y) for y in Y)



def wsum_d_to_set(X, X_counts, Y, d):
    return sum(X_counts[x] * d_to_set(x, Y, d) for x in X)



def dist(vec1, vec2, d, top_n=None):
    X_counts = vec1.value_counts()
    X = X_counts[:top_n].keys().to_list()

    Y_counts = vec2.value_counts()
    Y = Y_counts[:top_n].keys().to_list()

    wsum_X = wsum_d_to_set(X, X_counts, Y, d)
    wsum_Y = wsum_d_to_set(Y, Y_counts, X, d)

    return (wsum_X + wsum_Y) / (len(vec1) + len(vec2))



def compare_ips(df_ip1, df_ip2, text_features, cat_features, top_n):
    text_dist_avgs = [
        dist(df_ip1[feature], df_ip2[feature], lev_dist, top_n) 
        for feature in text_features
    ]

    cat_diff = [
        df_ip1.iloc[0][cat_feat] == df_ip2.iloc[0][cat_feat] 
        for cat_feat in cat_features
    ]

    dt_pval = ss.ks_2samp(df_ip1.dt, df_ip2.dt)[1]

    return text_dist_avgs + cat_diff + [dt_pval]



def construct_contrastive_dataset(df, bots, text_features, categorical_features, top_n=None):
    pairs = []

    for (i, bot1), (j, bot2) in product(enumerate(bots), enumerate(bots)):
        label = (i == j)
        for ip1, ip2 in product(bot1, bot2):
            df_ip1 = df[df.src == ip1]
            df_ip2 = df[df.src == ip2]
            contrastive_features = compare_ips(
                df_ip1, df_ip2, text_features, categorical_features, top_n)
            pairs += [contrastive_features + [label]]

    all_features = text_features + categorical_features + ['dt', 'label']
    df_contrastive = pd.DataFrame(pairs, columns=all_features)

    return df_contrastive


def contrast_ip(df, ip_baseline, ips_compare, text_features, categorical_features, top_n=None):
    pairs = []
    df_ip_base = df[df.src == ip_baseline]

    for ip_comp in ips_compare:
        df_ip_comp = df[df.src == ip_comp]
        contrastive_features = compare_ips(
            df_ip_base, df_ip_comp, text_features, categorical_features, top_n)
        pairs += [[ip_comp] + contrastive_features]

    all_features = ['src'] + text_features + categorical_features + ['dt']
    df_contrastive = pd.DataFrame(pairs, columns=all_features)

    return df_contrastive



def compare_ip_to_bot(df, ip, bot, text_features, categorical_features):
    xs = []
    df_ip = df[df.src == ip]
    for ip_cmp in bot:
        df_ip_cmp = df[df.src == ip_cmp]
        x = compare_ips(
            df_ip, 
            df_ip_cmp, 
            text_features, 
            categorical_features, 
            top_n=3)
        xs += [x]
    x_mean = np.array(xs).mean(axis=0)
    return x_mean



def compare_ips_to_bot(df, ips, bot, text_features, categorical_features):
    return [
        compare_ip_to_bot(
            df,
            ip, 
            bot, 
            text_features, 
            categorical_features) 
        for ip in ips
    ]



def sample_ips_train(bots, bot_idx, frac):
    bot = bots[bot_idx]
    n_train = int(np.ceil(frac * len(bot)))

    bot_train = list(np.random.choice(bot, n_train, replace=False))
    not_train = []
    for i, bot in enumerate(bots):
        if i == bot_idx:
            continue
        not_train += list(np.random.choice(bot, int(np.ceil(n_train/4))))

    return bot_train, not_train



def get_ips_test(df, bot_train, not_train):
    ips_train = bot_train + not_train
    ips_test = df.src[~np.isin(df.src, ips_train)].unique()
    return ips_test



def get_ip_labels(ips, bot):
    return [(ip in bot) for ip in ips]



def make_train_set(df, bots, bot_idx, frac, text_features, categorical_features):
    bot = bots[bot_idx]
    
    bot_train, not_train = sample_ips_train(bots, bot_idx, frac)
    ips_train = bot_train + not_train

    X = compare_ips_to_bot(
        df,
        ips_train, 
        bot_train, 
        text_features, 
        categorical_features)

    y = get_ip_labels(ips_train, bot)
    
    return bot_train, not_train, X, y



def make_test_set(df, bot, bot_train, not_train, X, y, text_features, categorical_features):
    ips_test = get_ips_test(df, bot_train, not_train)

    X = compare_ips_to_bot(
        df,
        ips_test, bot_train, 
        text_features, 
        categorical_features)

    y = get_ip_labels(ips_test, bot)

    return X, y



def train_test(df, bots, bot_idx, frac, text_features, categorical_features):
    bot = bots[bot_idx]
    bot_train, not_train, X_train, y_train = \
        make_train_set(
            df,
            bots, 
            bot_idx, 
            frac, 
            text_features, 
            categorical_features)

    clf = GradientBoostingClassifier().fit(X_train, y_train)

    X_test, y_test = \
        make_test_set(
            df, 
            bot, 
            bot_train, 
            not_train, 
            X_train, 
            y_train,
            text_features, 
            categorical_features)

    y_hats = np.array(clf.predict(X_test))
    y_test = np.array(y_test)
        
    return confusion_matrix(y_test, y_hats)