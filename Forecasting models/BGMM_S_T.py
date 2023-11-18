# coding: utf-8
import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from numpy.random import randn
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import inv as inv, det
from numpy.linalg import solve as solve
from scipy.linalg import null_space
from scipy.stats import invwishart
import time
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import properscoring as ps
import pickle


# from line_profiler import LineProfiler
# %load_ext line_profiler

def Sigma_Block(Sigma, con):
    dim2 = len(con)
    Sigma22 = np.eye(dim2)
    # construct Sigma22
    for i in range(dim2):
        for j in range(dim2):
            Sigma22[i, j] = Sigma[con[i], con[j]]
    return Sigma22

def Sample_mu_Sigma(X, lambda_0, mu_0, nu_0, Psi_0):
    N, L = X.shape
    X_bar = np.mean(X, axis=0)
    lambda_hyper = N + lambda_0
    mu_hyper = (N * X_bar + lambda_0 * mu_0) / lambda_hyper
    bar_d = np.repeat(X_bar.reshape(1, L), repeats=N, axis=0)
    S = (N - 1) * np.cov((X - bar_d).T)
    Psi_hyper = Psi_0 + S + N * lambda_0 / (N + lambda_0) * np.outer(X_bar - mu_0, X_bar - mu_0)
    nu_hyper = N + nu_0
    Sigma = invwishart.rvs(df=nu_hyper, scale=Psi_hyper)
    mu = mvnrnd(mu_hyper, Sigma / lambda_hyper)
    return mu, Sigma

def Corr(cov):
    '''
    Covariance to Correlation
    '''
    p = len(cov)
    e = np.eye(p)
    variance = e * cov
    v = np.power(variance, 0.5)
    I = np.linalg.inv(v)
    corr = I.dot(cov).dot(I)
    return corr

def imputation(mu, Sigma, cholSigma, G, r):
    y = cholSigma @ randn(mu.shape[0]) + mu
    y = y[:, np.newaxis]
    alpha = solve(G @ Sigma @ G.T, r - G @ y)
    x = (y + Sigma @ G.T @ alpha).T
    return x

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_mae(var, var_hat):
    return np.sum(np.abs(var - var_hat)) / var.shape[0]

with open('Train_BGMM_S_T.npy', 'rb') as dat:
    dat = np.load(dat)
mean_vec = dat[:,:-1].mean(axis = 0)
std_vec = dat[:,:-1].std(axis = 0)
obs_seq_train = (dat[:,:-1] - mean_vec) / std_vec
veh_id = dat[:,-1]-6

K = 10
N, link_num = obs_seq_train.shape
k_list = list(range(K))

epoach = 5000
flag = 0

HN = 16

# Store Bags
mu_K = np.zeros((K, link_num))
mu_store = np.zeros((epoach, K, link_num))
Sigma_K = np.zeros((K, link_num, link_num))
Sigma_store = np.zeros((epoach, K, link_num, link_num))
pi_store = np.zeros((epoach, HN, K))

# Initialization
pi = np.ones((HN, K))
pi = pi / K
Nk = np.zeros(K, dtype='int32')
z = np.random.choice(k_list, N, p = pi[0])

# Hyperparameters
alpha_0 = np.array([0.2 * K])
lambda_0 = 2
mu_0 = obs_seq_train.mean(axis = 0)
nu_0 = link_num + 2
Psi_0 = 0.01 * np.eye(link_num)
invSigma = np.zeros(Sigma_K.shape)
chol = np.zeros(Sigma_K.shape)
logdet = np.zeros(K)
logpi = np.zeros((HN, K))
start = time.time()

for it in range(epoach):

    # Update mu and sigma
    for k in k_list:
        Xk = obs_seq_train[np.where(z == k)]
        if Xk.shape[0] > 1:
            mu_K[k, :], Sigma_K[k, :, :] = Sample_mu_Sigma(Xk, lambda_0, mu_0, nu_0, Psi_0)
            mu_store[it, k, :], Sigma_store[it, k, :, :] = mu_K[k, :], Sigma_K[k, :, :]

    # Update pi
    for ia in range(HN):
        Z_interval = z[np.where(veh_id == ia)]
        for k in k_list:
            Nk[k] = len(Z_interval[np.where(Z_interval == k)])
        alpha = alpha_0 + Nk
        pi[ia] = np.random.dirichlet(alpha)
        pi_store[it, ia, :] = pi[ia]

    for k in range(K):
        invSigma[k,:,:] = inv(Sigma_K[k, :, :])
        chol[k, :, :] = np.linalg.cholesky(Sigma_K[k,:,:])
        logdet[k] = - np.sum(np.log(np.diag(chol[k, :, :])))
        for ia in range(HN):
            logpi[ia, k] = np.log(pi[ia, k])

    # Update hidden states zi
    for i in range(N):
        pzN = np.ones(K)
        for k in range(K):
            Apart = (obs_seq_train[i, :] - mu_K[k, :])
            pzN[k] = - 0.5 * Apart @ invSigma[k,:,:] @ Apart.T + logdet[k] + logpi[int(veh_id[i]), k]
        pz = np.exp(pzN - np.max(pzN))
        pz = pz/np.sum(pz)
        W = pz.cumsum()
        z[i] = W.searchsorted(np.random.uniform(0, W[-1]))

end = time.time()
print('Running time: %d seconds' % (end - start))

sample_num = 200
A_post = pi_store[-sample_num:]
mup_store = mu_store[-sample_num:]
Sigmap_store = Sigma_store[-sample_num:]

chol = {}
for k in range(K):
    chol['k'+str(k)] = []

for k in range(K):
    for i in range(Sigmap_store.shape[0]):
        chol['k'+str(k)].append(np.linalg.cholesky(Sigmap_store[i][k]))

all_predt = []
all_truet = []
all_predp = []
all_truep = []
rmset = []
mapet = []
rmsep = []
mapep = []
crpst = []
crpsp = []
maet = []
maep =[]
rmsett = []
maett = []
crpstt = []

with open("Test_BGMM_S_T", "rb") as fp:
    dat_set = pickle.load(fp)

for op in range(len(dat_set)):
    Bus_Data_Test = []
    for iu in dat_set[op]:
        dict_temp = {}
        dict_temp['con'] = np.array(iu['con'], dtype='int32')
        dict_temp['r'] = np.array(iu['obs'])
        dict_temp['pred_t'] = iu['pred_t']
        dict_temp['true_t'] = np.array(iu['true_t'])
        dict_temp['G'] = np.array(iu['G'], dtype='int32')
        dict_temp['Time'] = iu['Time']
        Bus_Data_Test.append(dict_temp)

    pred_t = []
    true_t = []
    pred_p = []
    true_p = []
    chol = np.zeros((K,link_num,link_num))
    ImpD = []
    Np = len(Bus_Data_Test)
    z = np.zeros(Np, dtype='int32')

    for it in range(sample_num):

        # get logpi
        pi_temp = A_post[it]
        for k in range(K):
            invSigma[k,:,:] = inv(Sigmap_store[it, k, :, :])
            chol[k, :, :] = np.linalg.cholesky(Sigmap_store[it, k,:,:])
            logdet[k] = - np.sum(np.log(np.diag(chol[k, :, :])))
            for h in range(HN):
                logpi[h, k] = np.log(pi[h, k])

        # Update hidden states zi
        for i in range(Np):
            pzN = np.ones(K)
            for k in range(K):
                Apart = (Bus_Data_Test[i]['r'] - mup_store[it, k, Bus_Data_Test[i]['con']])
                pzN[k] = - 0.5 * Apart @ invSigma[k,Bus_Data_Test[i]['con']][:,Bus_Data_Test[i]['con']] @ Apart.T + logdet[k] + logpi[int(veh_id[i]), k]
            pz = np.exp(pzN - np.max(pzN))
            pz = pz/np.sum(pz)
            W = pz.cumsum()
            z[i] = W.searchsorted(np.random.uniform(0, W[-1]))

        # Data imputation / Forecast
        ImpD_temp = []
        indext = []
        indexp = []
        for i in range(Np):
            if len(Bus_Data_Test[i]['con']) == link_num:
                temp = Bus_Data_Test[i]['r']
            else:
                temp = imputation(mup_store[it, z[i], :], Sigma_K[z[i], :, :], chol[z[i], :, :], Bus_Data_Test[i]['G'], Bus_Data_Test[i]['r'].reshape(-1,1))
            temp = temp.reshape(-1) * std_vec + mean_vec
            temp[np.where(temp<0)] = 0
            ImpD_temp.append(temp)
            if it > 100:
                indext.append(i)
                pred_t.append(list(temp[Bus_Data_Test[i]['pred_t']]))
        ImpD.append(ImpD_temp)
    rmset_temp = []
    mapet_temp = []
    rmsep_temp = []
    mapep_temp = []
    crpst_temp = []
    crpsp_temp = []
    maet_temp = []
    maep_temp = []
    rmsett_temp = []
    crpstt_temp = []
    maett_temp = []
    tt_true = []
    tt_pred = []
    for j in range(len(indext)):
        pred_tx = [pred_t[i] for i in range(len(pred_t)) if i % len(indext) == j]
        rmset_temp.append(compute_rmse(Bus_Data_Test[indext[j]]['true_t'],np.array(pred_tx).mean(axis=0)))
        mapet_temp.append(compute_mape(Bus_Data_Test[indext[j]]['true_t'],np.array(pred_tx).mean(axis=0)))
        maet_temp.append(compute_mae(Bus_Data_Test[indext[j]]['true_t'],np.array(pred_tx).mean(axis=0)))
        tt_pred.append(np.array(pred_tx).mean(axis=0).sum())
        tt_true.append(Bus_Data_Test[indext[j]]['true_t'].sum())
        tx = np.array(pred_tx)
        crpstt_temp.append(ps.crps_ensemble(Bus_Data_Test[indext[j]]['true_t'].sum(), tx[:,:].sum(axis=1)))
        for y in range(len(Bus_Data_Test[indext[j]]['true_t'])):
            crpst_temp.append(ps.crps_ensemble(Bus_Data_Test[indext[j]]['true_t'][y], tx[:,y]))
    rmsett_temp.append(compute_rmse(np.array(tt_true),np.array(tt_pred)))
    maett_temp.append(compute_mae(np.array(tt_true),np.array(tt_pred)))
    rmset.append(rmset_temp)
    rmsett.append(rmsett_temp)
    mapet.append(mapet_temp)
    crpst.append(crpst_temp)
    crpstt.append(crpstt_temp)
    maet.append(maet_temp)
    maett.append(maett_temp)