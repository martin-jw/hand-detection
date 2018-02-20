from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from time import time

parameters = [{
	'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-4, 1e-2], 'svc__C': [1, 10, 100, 1000]
}, {
	'svc__kernel': ['linear'], 'svc__C': [1, 10, 100, 1000]
}]

def create_classifier(X, y, params=parameters): 
	pline = Pipeline([('transf', preprocessing.StandardScaler()), ('svc', svm.SVC(probability=True))])

	grid_search = GridSearchCV(pline, params, n_jobs=-1, verbose=1)

	print("Creating classifier: Performing grid search...")

	t0 = time()
	grid_search.fit(X, y)
	print("Done in %0.3fs" % (time() - t0))
	print()

	print("Best score: %0.3fs" % grid_search.best_score_) 
	print("parameters selected:")
	best_params = grid_search.best_estimator_.get_params()
	for param_name in sorted(best_params.keys()):
		print("\t%s: %r" % (param_name, best_params[param_name]))

	pline.set_params(**best_params)
	return pline
