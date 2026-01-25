
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import beta
from deep_eknn import EKNN
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize_scalar
from laplace import Laplace
import torch
from NN import PyTorchMLP, PyTorchMLPDropout, PyTorchMLPDropConnect
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import GPy
import pymc as pm
import numpy as np

def BDL(X, y, X_test):

    model = PyTorchMLP(input_dim=X.shape[1], hidden_dim_1=100, hidden_dim_2=20, output_dim=len(set(y)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    for epoch in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='diag')

    la.fit(loader)
    la.optimize_prior_precision(method='marglik')

    p_pred = la.predictive_samples(X_test_tensor, n_samples=100).numpy().astype(float)
    preds = la(X_test_tensor).numpy().astype(float)

    ep = np.var(p_pred[:,:,0], axis=0)

    return ep, p_pred, preds

def likelihood(X, y, X_test, n_neighbors=10):

    # Load model for classification
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

    # Fit model according to the dataset
    classifier.fit(X, y)
    preds = classifier.predict_proba(X_test)
    
    epistemic_list = []

    # Compute epistemic uncertainty
    for i in range(X_test.shape[0]):
        dist, indices = classifier.kneighbors(np.array([X_test[i]]), min(X.shape[0], 200))

        closest = (dist < 0.3)[0]

        if dist[0, closest].shape[0] == 0:
            epistemic, _ = compute_epistemic([], indices[0, 0:1], np.array(y, dtype=int))
        else:
            epistemic, _ = compute_epistemic(dist[0, closest], indices[0, closest], np.array(y, dtype=int))
        epistemic_list.append(epistemic)

    ep = np.array(epistemic_list)

    p_pred_0 = []
    for i in range(200):
        sample = np.random.normal(preds[:,0], np.sqrt(ep))
        sample = np.clip(sample, 0, 1)
        p_pred_0.append(sample)
    p_pred_0 = np.array(p_pred_0)
    p_pred = np.array([p_pred_0, 1-p_pred_0]).transpose(1, 2, 0)

    return ep, p_pred, preds

# Compute epistemic uncertainty
def compute_epistemic(dist, indices, y):
    nb_classes = 2
    res = np.zeros(nb_classes)

    for i in range(indices.shape[0]):
        res[y[indices[i]]] += 1

    p = res[0]
    n = res[1]

    opt = minimize_scalar(f_objective_1, bounds=(0, 1), method='bounded', args=(p, n))
    pl1 = opt.x

    opt = minimize_scalar(f_objective_2, bounds=(0, 1), method='bounded', args=(p, n))
    pl2 = 1 - opt.x

    ue = min(pl1, pl2) - 0.5
    ua = 1 - max(pl1, pl2)

    return ue, ua

# Objective fuction used to compute epistemic uncertainty
def f_objective_1(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 2 * theta - 1

    res = min(left, right)

    return -res

# Objective fuction used to compute epistemic uncertainty
def f_objective_2(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 1 - (2 * theta)
        
    res = min(left, right)

    return -res


def beta_kde(data, grid=None, h=None):
    data = np.asarray(data).ravel()
    if grid is None:
        grid = np.linspace(0.0, 1.0, 200)
    grid = np.asarray(grid)
    n = data.size
    if n == 0:
        return grid, np.zeros_like(grid)
    if h is None:
        h = np.clip(n**(-1/5), 1e-3, 0.5)
    c = max(1e-12, 1.0 / (h**2) - 1.0)
    a = data * c + 1.0
    b = (1.0 - data) * c + 1.0
    eps = 1e-12
    grid_clipped = np.clip(grid, eps, 1.0 - eps)
    pdf_vals = beta.pdf(grid_clipped[:, None], a[None, :], b[None, :])
    density = pdf_vals.mean(axis=1)
    return grid, density

def randomForest(X, y, X_test, nb_estimators=200):
    model = RandomForestClassifier(n_estimators=nb_estimators, min_samples_leaf=5)

    model.fit(X, y)
    preds = np.array(model.predict_proba(X_test))

    # TREE
    unc_tree = []
    p_pred = []
    for tree in model.estimators_:
        pred_t = tree.predict_proba(X_test)
        p_pred.append(pred_t)
        unc_tree.append(-np.sum(pred_t * np.log2(pred_t + 1e-10), axis=1))

    p_pred = np.array(p_pred)
    ep = np.var(p_pred[:,:,0], axis=0)
    return ep, p_pred, preds

def EvKNN(X, y, X_test):
    model = EKNN(2, 12)

    model.fit(X, y)
    preds = np.array(model.predict_proba(X_test))

    _, ep = model.get_uncertainties(X_test)
    
    p_pred_0 = []
    for i in range(200):
        sample = np.random.normal(preds[:,0], np.sqrt(ep))
        sample = np.clip(sample, 0, 1)
        p_pred_0.append(sample)
    p_pred_0 = np.array(p_pred_0)
    p_pred = np.array([p_pred_0, 1-p_pred_0]).transpose(1, 2, 0)

    return ep, p_pred, preds

def gaussianProcess(X, y, X_test):
    kernel = GPy.kern.RBF(input_dim=X.shape[1])

    y_train_2 = np.zeros((X.shape[0], 1))
    y_train_2[:,0] = y

    model = GPy.models.GPClassification(X, y_train_2, kernel=kernel)
    model.optimize()
    mean_f, var_f = model._raw_predict(X_test)

    var_f = np.clip(var_f, 1e-12, None)
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(size=(X_test.shape[0], 300))
    f_samples = mean_f.ravel()[:, None] + np.sqrt(var_f).ravel()[:, None] * eps
    p_samples = 1.0 / (1.0 + np.exp(-f_samples))
    p_var = p_samples.var(axis=1).reshape(-1, 1)

    p_pred = np.zeros((p_samples.shape[1], p_samples.shape[0], 2))
    p_pred[:,:,1] = p_samples.T
    p_pred[:,:,0] = 1 - p_samples.T
    preds = np.mean(p_pred, axis=0)
    ep = p_var[:,0]

    return ep, p_pred, preds

def centroids_uncertainties(X, y, X_test, length_scale=10):
    unique_classes = np.unique(y)
    centroids = {}
    for cls in unique_classes:
        cls_positions = X[y == cls]
        centroids[cls] = np.mean(cls_positions, axis=0)

    size = len(centroids)

    uncertainties = []
    for c in centroids:
        l2 = np.linalg.norm(X_test - centroids[c], ord=2, axis=1)
        l2 = (1/size) * (l2**2)
        l2 = l2 / (2 * length_scale)
        l2 = np.exp(-l2)
        uncertainties.append(l2)
    
    preds = uncertainties / np.sum(uncertainties, axis=0)
    preds = np.array(preds).T

    ep = - np.log(np.max(uncertainties, axis=0))
    ep = ep / (ep + (10.0 * np.median(ep))) * 0.8

    p_pred_0 = []
    for i in range(200):
        sample = np.random.normal(preds[:,0], np.sqrt(ep))
        sample = np.clip(sample, 0, 1)
        p_pred_0.append(sample)
    p_pred_0 = np.array(p_pred_0)
    p_pred = np.array([p_pred_0, 1-p_pred_0]).transpose(1, 2, 0)

    return ep, p_pred, preds

def drop_connect(X, y, X_test, nb_estimators=100):   

    model = PyTorchMLPDropConnect(input_dim=X.shape[1], hidden_dim_1=100, hidden_dim_2=20, output_dim=len(set(y)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    for epoch in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.train() 
    p_preds = []

    with torch.no_grad():
        for _ in range(nb_estimators):
            p = model(X_test_tensor).unsqueeze(0)
            p_preds.append(torch.softmax(p, dim=2))

    p_pred = np.array(p_preds)[:,0,:,:]
    ep = np.var(p_pred[:,:,0], axis=0)
    preds=np.mean(p_pred, axis=0)

    return ep, p_pred, preds

def mc_dropout(X, y, X_test, nb_estimators=100):   

    model = PyTorchMLPDropout(input_dim=X.shape[1], hidden_dim_1=100, hidden_dim_2=20, output_dim=len(set(y)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    for epoch in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.train() 
    p_preds = []

    with torch.no_grad():
        for _ in range(nb_estimators):
            p = model(X_test_tensor).unsqueeze(0)
            p_preds.append(torch.softmax(p, dim=2))

    p_pred = np.array(p_preds)[:,0,:,:]
    ep = np.var(p_pred[:,:,0], axis=0)
    preds=np.mean(p_pred, axis=0)

    return ep, p_pred, preds

def deep_ensemble(X, y, X_test, nb_estimators=10):    
    models = []
    indices = np.arange(X.shape[0])

    preds = []

    for i in range(nb_estimators):
        models.append(MLPClassifier(hidden_layer_sizes=(100, 20), activation='relu', solver='adam', max_iter=500, random_state=i))
        models[i].fit(X[indices], y[indices])
        np.random.shuffle(indices)

        preds.append(models[i].predict_proba(X_test))

    p_pred = np.array(preds)
    ep = np.var(p_pred[:,:,0], axis=0)
    preds=np.mean(p_pred, axis=0)

    return ep, p_pred, preds

def bayesianLR(X, y, X_test, n_samp=1000):
    with pm.Model() as model:
        w = pm.Normal("w", mu=0, sigma=1, shape=X.shape[1])
        b = pm.Normal("b", mu=0, sigma=1)
        
        # Logits et probas
        logits = pm.math.dot(X, w) + b
        p = pm.Deterministic("p", pm.math.sigmoid(logits))
        
        # Observation
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
        
        # Inférence MCMC
        
        trace = pm.sample(n_samp, tune=n_samp, chains=2, target_accept=0.5, progressbar=True)

    logits_pred = trace.posterior["w"].values @ X_test.T + trace.posterior["b"].values[..., np.newaxis]
    p_pred = 1 / (1 + np.exp(-logits_pred))
    p_mean = p_pred.mean(axis=(0, 1))
    ep = p_pred.var(axis=(0, 1))
    preds = np.array([1- p_mean, p_mean]).T
    p_pred = 1 - p_pred.transpose(1, 2, 0)

    return ep, p_pred, preds