import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from numpy.random import randn
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from scipy.stats import invwishart
import time
import properscoring as ps
import pickle

# from line_profiler import LineProfiler
# %load_ext line_profiler

def Sigma_Block(Sigma, con):
    '''
    Sigma: np.array n*n
    var: list [1, 2]
    con: list [4, 5, 6]
    '''

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

def imputation(mu, Sigma, G, r):
    cholSigma = np.linalg.cholesky(Sigma)
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

def mnrnd(M, U, V):
    """
    Generate matrix normal distributed random matrix.
    M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
    """
    r, l = M.shape
    X0 = np.random.randn(r, l)
    P = np.linalg.cholesky(U + 0.00001 * np.eye(r))
    Q = np.linalg.cholesky(V + 0.00001 * np.eye(r))
    return M + P @ X0 @ Q.T

with open('Train_S_P.npy', 'rb') as dat:
    dat = np.load(dat)

obs_seq = (dat - dat.mean(axis = 0)) / dat.std(axis = 0)
mean_vec = dat.mean(axis = 0)
std_vec = dat.std(axis = 0)
obs_seq = (dat - mean_vec) / std_vec

K = 10
N, link_num = obs_seq.shape
k_list = list(range(K))

epoach = 5000
flag = 0

# Store Bags
mu_K = np.zeros((K, link_num))
mu_store = np.zeros((epoach, K, link_num))
Sigma_K = np.zeros((K, link_num, link_num))
Sigma_store = np.zeros((epoach, K, link_num, link_num))
B_K = np.zeros((K, link_num, link_num))
B_store = np.zeros((epoach, K, link_num, link_num))
A_store = np.zeros((epoach, K, K))

# Initialization
pi = np.ones((K, K))
pi_temp = np.ones((K, K))
pi = pi / np.sum(pi[0])
Nk = np.zeros(K, dtype='int32')
z = np.random.choice(k_list, N, p=pi[0])
A_count = np.zeros((K, K), dtype='int32')
A0_count = np.zeros(K, dtype='int32')
X_temp = np.zeros((N - 1, link_num))

# Hyperparameters
alpha_0 = np.array([0.1] * K)
lambda_0 = 0.1
mu_0 = obs_seq.mean(axis=0)
nu_0 = link_num + 2
Psi_0 = 0.01 * np.eye(link_num)
M_0 = np.zeros((link_num, link_num))
V_0 = 0.01 * np.eye(link_num)
invSigma = np.zeros(Sigma_K.shape)
chol = np.zeros(Sigma_K.shape)
logdet = np.zeros(K)
logpi = np.zeros((K, K))
start = time.time()

for it in range(epoach):
    # Update A
    for k in k_list:
        inv_V_0 = inv(V_0)
        V0 = inv(inv_V_0 + obs_seq[np.where(z == k)[0] - 1].T @ obs_seq[np.where(z == k)[0] - 1])
        M0 = (M_0 @ inv_V_0 + (obs_seq[np.where(z == k)] - mu_K[k]).T @ obs_seq[np.where(z == k)[0] - 1]) @ V0
        B_K[k] = mnrnd(M0, Sigma_K[k], V0)
        B_store[it, k] = B_K[k]

    # Update mu and sigma
    for k in k_list:
        Xk = obs_seq[np.where(z == k)] - obs_seq[np.where(z == k)[0] - 1] @ B_K[k].T
        if Xk.shape[0] > 1:
            mu_K[k, :], Sigma_K[k, :, :] = Sample_mu_Sigma(Xk, lambda_0, mu_0, nu_0, Psi_0)
            mu_store[it, k, :], Sigma_store[it, k, :, :] = mu_K[k, :], Sigma_K[k, :, :]

    # Update pi and pi0
    for i in range(N - 1):
        A_count[z[i], z[i + 1]] += 1
    for ik in k_list:
        for ij in k_list:
            Nk[ij] = A_count[ik, ij]
        alpha = alpha_0 + Nk
        pi_temp[ik, :] = np.random.dirichlet(alpha)
    A_store[it, :, :] = pi_temp

    for k in range(K):
        invSigma[k, :, :] = inv(Sigma_K[k, :, :])
        chol[k, :, :] = np.linalg.cholesky(Sigma_K[k, :, :])
        logdet[k] = - np.sum(np.log(np.diag(chol[k, :, :])))
        for ik in range(K):
            logpi[ik, k] = np.log(pi_temp[ik, k])

    for i in range(N):
        A0_count[z[i]] += 1
    pi0 = np.array(A0_count) / np.array(A0_count).sum()

    # Upda hidden state zi and z0
    W = pi0.cumsum()
    z[0] = W.searchsorted(np.random.uniform(0, W[-1]))

    for i in range(1, N):
        pzN = np.ones(K)
        for k in range(K):
            Apart = (obs_seq[i, :] - B_K[k] @ obs_seq[i - 1, :] - mu_K[k, :])
            if i == N - 1:
                pzN[k] = - 0.5 * Apart @ invSigma[k, :, :] @ Apart.T + logdet[k] + logpi[z[i - 1], k]
            else:
                pzN[k] = - 0.5 * Apart @ invSigma[k, :, :] @ Apart.T + logdet[k] + logpi[z[i - 1], k] + logpi[
                    k, z[i + 1]]
        pz = np.exp(pzN - np.max(pzN))
        pz = pz / np.sum(pz)
        W = pz.cumsum()
        z[i] = W.searchsorted(np.random.uniform(0, W[-1]))

end = time.time()
print('Running time: %d seconds' % (end - start))

sample_num = 200
A_post = A_store[-sample_num:]
B_post = B_store[-sample_num:]
mup_store = mu_store[-sample_num:]
Sigmap_store = Sigma_store[-sample_num:]

chol = {}
for k in range(K):
    chol['k'+str(k)] = []

for k in range(K):
    for i in range(Sigmap_store.shape[0]):
        chol['k'+str(k)].append(np.linalg.cholesky(Sigmap_store[i][k]))

with open("Test_S_P", "rb") as fp:
    dat_set = pickle.load(fp)

all_predp = []
all_truep = []
rmsep = []
mapep = []
crpsp = []
maep =[]

for op in range(len(dat_set)):
    Bus_Data_Test = []
    for iu in dat_set[op]:
        dict_temp = {}
        dict_temp['con'] = np.array(iu['con'], dtype='int32')
        dict_temp['r'] = np.array(iu['obs'])
        dict_temp['rp'] = np.array(iu['obsp'])
        dict_temp['pred_p'] = iu['pred_p']
        dict_temp['true_p'] = np.array(iu['true_p'])
        dict_temp['G'] = np.array(iu['G'], dtype='int32')
        dict_temp['Time'] = iu['Time']
        dict_temp['full'] = iu['full']
        Bus_Data_Test.append(dict_temp)

    pred_p = []
    true_p = []
    chol = np.zeros((K, link_num, link_num))
    ImpD = []
    Np = len(Bus_Data_Test)
    y_jm = np.zeros(Np, link_num)
    y_jm[0] = Bus_Data_Test[0]['full']

    for it in range(sample_num):

        # get logpi
        pi_temp = A_post[it]
        for k in range(K):
            invSigma[k,:,:] = inv(Sigmap_store[it, k, :, :])
            chol[k, :, :] = np.linalg.cholesky(Sigmap_store[it, k,:,:])
            logdet[k] = - np.sum(np.log(np.diag(chol[k, :, :])))
            for ik in range(K):
                logpi[ik, k] = np.log(pi_temp[ik, k])

        # Update hidden states zi
        z = np.zeros(Np, dtype='int32')
        for i in range(1,Np):
            pzN = np.ones(K)
            for k in range(K):
                yi = (B_post[it, k] @ y_jm[i-1].T).T
                if i == Np - 1:
                    Apart = (Bus_Data_Test[i]['r'] - yi[Bus_Data_Test[i]['con']] - mup_store[it, k, Bus_Data_Test[i]['con']])
                    pzN[k] = - 0.5 * Apart @ invSigma[k, Bus_Data_Test[i]['con']][:,Bus_Data_Test[i]['con']] @ Apart.T + logdet[k] + logpi[z[i - 1], k]
                else:
                    Apart = (Bus_Data_Test[i]['r'] - yi[Bus_Data_Test[i]['con']] - mup_store[it, k, Bus_Data_Test[i]['con']])
                    pzN[k] = - 0.5 * Apart @ invSigma[k, Bus_Data_Test[i]['con']][:,Bus_Data_Test[i]['con']] @ Apart.T + logdet[k] + logpi[z[i - 1], k] + logpi[k, z[i + 1]]
            pz = np.exp(pzN - np.max(pzN))
            pz = pz/np.sum(pz)
            W = pz.cumsum()
            z[i] = W.searchsorted(np.random.uniform(0, W[-1]))

        # Data imputation / Forecast
        ImpD_temp = []
        indext = []
        indexp = []

        for i in range(1, Np - 1):
            m_j = B_post[it,z[i]] @ y_jm[i-1] + mup_store[it, z[i]]
            m_jp1 = B_post[it,z[i+1]] @ m_j + mup_store[it, z[i+1]]
            m = np.vstack([m_j, m_jp1])

            L_ff = Sigmap_store[it, z[i]]
            L_fo = L_ff @ B_post[it,z[i+1]].T
            L_of = B_post[it,z[i+1]] @ L_ff
            L_oo = B_post[it,z[i+1]] @ L_ff @ B_post[it,z[i+1]].T + Sigmap_store[it, z[i+1]]
            L = np.block([[L_ff, L_fo], [L_of, L_oo]])
            rec = Bus_Data_Test[i]['rp']
            y_jm1 = imputation(m, L, Bus_Data_Test[i]['G'], rec.reshape(-1,1))
            temp = y_jm1[:link_num].reshape(-1) * std_vec + mean_vec
            temp[np.where(temp < 0)] = 0
            ImpD_temp.append(temp)
            y_jm[i] = y_jm1[:link_num]
            if it > 100:
                indexp.append(i)
                pred_p.append(list(temp[Bus_Data_Test[i]['pred_p']]))
        ImpD.append(ImpD_temp)
    rmsep_temp = []
    mapep_temp = []
    crpsp_temp = []
    maep_temp = []
    for j in range(len(indexp)):
        pred_px = [pred_p[i] for i in range(len(pred_p)) if i%len(indexp) == j]
        rmsep_temp.append(compute_rmse(Bus_Data_Test[indexp[j]]['true_p'],np.array(pred_px).mean(axis=0)))
        mapep_temp.append(compute_mape(Bus_Data_Test[indexp[j]]['true_p'],np.array(pred_px).mean(axis=0)))
        maep_temp.append(compute_mae(Bus_Data_Test[indexp[j]]['true_p'],np.array(pred_px).mean(axis=0)))
        px = np.array(pred_px)
        for y in range(len(Bus_Data_Test[indexp[j]]['true_p'])):
            crpsp_temp.append(ps.crps_ensemble(Bus_Data_Test[indexp[j]]['true_p'][y], px[:,y]))
    rmsep.append(rmsep_temp)
    mapep.append(mapep_temp)
    crpsp.append(crpsp_temp)