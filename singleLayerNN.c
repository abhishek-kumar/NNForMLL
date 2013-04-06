
#include "nn.h"
#include "io.h"
#include "parameters.h"
#include "singleLayerNN.h"

// ctors
singleLayerNN::singleLayerNN(io & fileio, int numFeatures, int numHiddenUnits, int numTags) : 
				p(numFeatures), d(numHiddenUnits), k(numTags),
				linearity(0.0), wopt(0), counter(0), xtr(fileio.xtr), ytr(fileio.ytr), m(xtr.size()) { }

singleLayerNN::singleLayerNN(data_t& xtrain, data_t& ytrain, int numFeatures, int numHiddenUnits, 
	int numTags, floatnumber regularizationStrength) : 
			p(numFeatures), d(numHiddenUnits), k(numTags), C(regularizationStrength),
			linearity(0.0), wopt(0), counter(0), xtr(xtrain), ytr(ytrain), m(xtr.size()) { }

void singleLayerNN::fit() {

	// Initialize parameters for LBFGS
	if(wopt) delete wopt;
	wopt = new parameters(p,d,k,true);
	/*
	cout << "SingleLayerNN: Initialized weights as:\n";
	for(int i=0; i < wopt->N; ++i)
		cout << "(" << i+i << ", " << ((*wopt)[i]) << "), ";
	cout << endl;
	*/
	/*
	for(int i=0, f=wopt->getvector()+wopt->layer1N+wopt->layer2N; 
	    i<wopt->layer3N; 
			++i,++f)
			*f = 0.0;
	*/
	floatnumber bestloss = 1e+5;
	lbfgs_parameter_t param; int ret=0; lbfgs_parameter_init(&param);
	param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;

	// Run the LBFGS training process
	counter = 0; // count of iterations done so far 

	Compatibility::model = this;
	ret = lbfgs(wopt->N, wopt->getvector(), &bestloss, Compatibility::evaluate, Compatibility::progress, &xtr, &param);

	/*
	cout << "SingleLayerNN: Learned weights as:\n";
	for(int i=0; i < wopt->N; ++i)
		cout << "(" << i+i << ", " << ((*wopt)[i]) << "), ";
	cout << endl;
  */
	/* Report the result. */
  printf("\tL-BFGS optimization terminated with status code = %d", ret);
  printf("; bestLoss = %f\n", bestloss);

}

void singleLayerNN::train(int lowerLimit, int upperLimit, int stepSize, int cvFolds) {
	// cross validate to find regularization strength
	int indexes[m]; for(int i=0; i<m; ++i) indexes[i] = i; randomShuffle(indexes, m); 
	
	floatnumber bestC=0.0, lastNLL=1e10, lasttolastNLL=1e10;
	floatnumber bestNllMean=1e10, bestNllStdev;
	floatnumber bestHlMean, bestHlStdev, bestSlMean, bestSlStdev, bestRlMean, bestRlStdev, bestNrlMean, 
			bestNrlStdev, bestOeMean, bestOeStdev, bestAvprecMean, bestAvprecStdev;

	// For each value of C in grid
	for(int cc = lowerLimit; cc < upperLimit; cc += stepSize)
	{
		floatnumber C = pow(2,cc);
		record_t cvLossesNll, cvLossesHl, cvLossesSl, cvLossesRl, cvLossesNrl, cvLossesOe, cvLossesAvprec;

		// For each fold
		for(int cv=0; cv<cvFolds; ++cv)
		{
			int testsetstart = cv*float(m)/cvFolds;
			int testsetend = testsetstart+float(m)/cvFolds;
			data_t cvxtrain, cvytrain; cvxtrain.clear(); cvytrain.clear();
			data_t cvxtest, cvytest; cvxtest.clear(); cvytest.clear();
			for(int i=0; i<testsetstart; ++i)
			{
				cvxtrain.push_back(xtr[indexes[i]]);
				cvytrain.push_back(ytr[indexes[i]]);
			}
			for(int i=testsetstart; i<testsetend; ++i)
			{
				cvxtest.push_back(xtr[indexes[i]]);
				cvytest.push_back(ytr[indexes[i]]);
			}
			for(int i=testsetend; i<m; ++i)
			{
				cvxtrain.push_back(xtr[indexes[i]]);
				cvytrain.push_back(ytr[indexes[i]]);
			}

			singleLayerNN tempModel(cvxtrain, cvytrain, p, d, k, C);
			tempModel.fit();
			error_t loss = tempModel.test(cvxtest, cvytest);
			cvLossesNll.push_back(loss.nll); cvLossesHl.push_back(loss.hl); cvLossesSl.push_back(loss.sl); 
			cvLossesRl.push_back(loss.rl); cvLossesNrl.push_back(loss.nrl); cvLossesOe.push_back(loss.oe); cvLossesAvprec.push_back(loss.avprec);
		}

		floatnumber nllMean, nllStdev, hlMean, hlStdev, slMean, slStdev, rlMean, rlStdev, nrlMean, 
				nrlStdev, oeMean, oeStdev, avprecMean, avprecStdev;
		cout << "Regularization Parameter Training | C: 2^" << cc;
		meanAndStdev(cvLossesNll, nllMean, nllStdev);
		cout << ", NLL: " << fixed << nllMean << " +- " << nllStdev;
		meanAndStdev(cvLossesHl, hlMean, hlStdev);
		cout << ", HL: " << fixed << hlMean << " +- " << hlStdev;
		meanAndStdev(cvLossesSl, slMean, slStdev);
		cout << ", SL: " << fixed << slMean << " +- " << slStdev;
		meanAndStdev(cvLossesRl, rlMean, rlStdev);
		cout << ", RL: " << fixed << rlMean << " +- " << rlStdev;
		meanAndStdev(cvLossesNrl, nrlMean, nrlStdev);
		cout << ", NRL: " << fixed << nrlMean << " +- " << nrlStdev;
		meanAndStdev(cvLossesOe, oeMean, oeStdev);
		cout << ", OE: " << fixed << oeMean << " +- " << oeStdev;
		meanAndStdev(cvLossesAvprec, avprecMean, avprecStdev);
		cout << ", AVPREC: " << fixed << avprecMean << " +- " << avprecStdev;
		cout << endl;

		if(nllMean<bestNllMean)
		{
			bestC = C;
			bestNllMean = nllMean; bestNllStdev = nllStdev;
			bestHlMean = hlMean; bestHlStdev = hlStdev;
			bestSlMean = slMean; bestSlStdev = slStdev;
			bestRlMean = rlMean; bestRlStdev = rlStdev;
			bestNrlMean = nrlMean; bestNrlStdev = nrlStdev;
			bestOeMean = oeMean; bestOeStdev = oeStdev;
			bestAvprecMean = avprecMean; bestAvprecStdev = avprecStdev;
		}
		if( (lasttolastNLL+0.01) < lastNLL && (lastNLL+0.01) < nllMean) // we're doing worse down this path. stop.
		{
			cout << "Regularization Parameter Training | Further parameter checking is futile, losses seem to increase.\n"
			     << "Halting at bestC: " << bestC << " (currently evaluated C: " << C << ")\n";
			break;
		}
		lasttolastNLL = lastNLL; lastNLL = nllMean;
	}
	cout << "Regularization Parameter Training | Training complete. Best C: " 
	     << bestC << " || NLL=" << bestNllMean << endl;
	
	this->C = bestC;
	fit();

	ofstream resultsfile; resultsfile.open("resultstrain_sl.txt", ios::app);
	resultsfile << endl << "Results for training set" << endl;
	resultsfile << d << "\t" << bestC << "\t" << bestNllMean << " +- " << bestNllStdev << "\t" 
	            << bestHlMean << " +- " << bestHlStdev << "\t"
	            << bestSlMean << " +- " << bestSlStdev << "\t"
	            << bestRlMean << " +- " << bestRlStdev << "\t"
	            << bestNrlMean << " +- " << bestNrlStdev << "\t"
	            << bestOeMean << " +- " << bestOeStdev << "\t"
	            << bestAvprecMean << "\t" << bestAvprecStdev << endl;
	resultsfile.close();
	return;
}

inline void singleLayerNN::calculateJacobian(	record_t const & x,
                              					record_t const & y,
												const floatnumber *y_hata, const floatnumber *y_hat,
												const floatnumber *ha, const floatnumber *h,
												parameters const & w,
												parameters & jacobian) {
	// Calculate deltaLoss / deltaActivation for both levels
	floatnumber dy[k], dh[d];
	for(int j=0; j<k; ++j)
	{
		floatnumber yi = 2*y[j] - 1; // convert to {-1,+1}
		floatnumber minusyi = -yi;
		dy[j] = minusyi*exp(minusyi*y_hata[j]) / (1.0+exp(minusyi*y_hata[j]));
		dy[j] -= linearity*yi;
	}

	for(int j=0; j<d; ++j)
	{
		floatnumber delta=0.0;
		for(int kk=0; kk<k; ++kk)
			delta += dy[kk]*w.val(1,j,kk);
		dh[j] = delta*(1.0 + linearity - pow(tanh(ha[j]),2) );
	}

	// Calculate gradients (w/o regularization)
	for(int i=0; i<p; ++i)
		for(int j=0; j<d; ++j)
			jacobian.val(0,i,j) += dh[j]*x[i];
	for(int j=0; j<d; ++j)
		jacobian.val(0,p,j) += dh[j];

	for(int i=0; i<d; ++i)
		for(int j=0;j<k; ++j)
			jacobian.val(1,i,j) += dy[j]*h[i];
	for(int j=0; j<k; ++j)
		jacobian.val(1,d,j) += dy[j];
};

void singleLayerNN::forwardPropagate(	record_t const & x, 
                    				  	parameters const & w, 
										floatnumber *y_hata_, 
										floatnumber *y_hat_, 
										floatnumber *ha_, 
										floatnumber *h_) {
	// Layer 0
	for(int j=0; j<d; ++j)
		ha_[j] = w.val(0,p,j);
	for(int i=0; i<p; ++i)
		for(int j=0; j<d; ++j)
			ha_[j] += x[i]*w.val(0,i,j);
	for(int j=0; j<d; ++j)
		h_[j] = tanh(ha_[j]) + linearity*ha_[j];
	
	// Layer 1
	for(int j=0; j<k; ++j)
		y_hata_[j] = w.val(1,d,j);
	for(int i=0; i<d; ++i)
		for(int j=0; j<k; ++j)
			y_hata_[j] += h_[i]*w.val(1,i,j);
	for(int j=0; j<k; ++j)
		y_hat_[j] = 1.0/(1.0+exp((-1)*y_hata_[j]));
}


error_t singleLayerNN::test(data_t xtest, data_t ytest) {
	int sz = xtest.size();
	floatnumber y_hata[k], y_hat[k], ha[d], h[d], curloss, hl;
	error_t loss = error_t();
	record_t xrecord, yrecord;

	for(int i = 0; i < sz; ++i)
	{
		xrecord = xtest[i]; yrecord = ytest[i];
		forwardPropagate(xrecord, *wopt, y_hata, y_hat, ha, h);
		calculateLosses(y_hata, y_hat, yrecord, *wopt, curloss, hl);
		loss.nll += curloss; loss.hl += hl; if(hl>0) ++(loss.sl);

		// Number of relevant tags (for normalized RL)
		floatnumber r=0; for(int temp=0; temp<k; ++temp) r+=yrecord[temp];

		// Ranking Loss
		floatnumber rl=0.0, rl2 =0.0;
		for(int pid=0; pid<k; ++pid)
			for(int nid=0; nid<k; ++nid)
				if(yrecord[pid] > 0.5 && yrecord[nid] < 0.5)
				{
					if(y_hat[pid]<y_hat[nid]) ++rl, ++rl2;
					if(y_hat[pid]==y_hat[nid]) rl+=0.5, ++rl2;
				}
		loss.rl += rl;
		loss.nrl += rl2 / float(r*(k-r));

		// One Error
		floatnumber ymax=0.0; int argmaxy = -1;
		for(int ii=0; ii<k; ++ii)
			if(y_hat[ii] > ymax)
			{
				ymax = y_hat[ii];
				argmaxy = ii;
			}
		if(yrecord[argmaxy] < 0.5) ++(loss.oe);

		// Average Precision
		floatnumber oe=0.0;
		for(int ii=0; ii<k; ++ii)
		{
			floatnumber rank=0.0; for(int iii=0; iii<k; ++iii) if(y_hat[iii] >= y_hat[ii]) ++rank;
			floatnumber count = 0.0; for(int iii=0; iii<k; ++iii) if(y_hat[iii] >= y_hat[ii]) count += yrecord[iii];
			oe += (yrecord[ii])*(count / rank);
		}
		loss.avprec += oe/float(r);
	}
	loss.hl /= sz*k; loss.sl /= sz, loss.rl /= sz; loss.nrl /= sz; loss.oe /= sz; loss.avprec /= sz;

	cout << "Debug print: ";
	loss.print();
	//cout << "Top Tags for interpretability: " << endl;
	//printTagCorrelations(*wopt, 6);
	return loss;
}

void singleLayerNN::calculateLosses(const floatnumber *y_hata,
                     				const floatnumber *y_hat,
									record_t const & y,
									parameters const & w,
									floatnumber & nll,
									floatnumber & hl) {
	// All losses without regularization
	nll = 0; hl = 0;

	for(int i=0; i<k; ++i)
	{
		floatnumber yi = 2*y[i] - 1; // convert to {-1,+1}
		floatnumber minusyi = (-1)*yi;
		nll += log(1.0+exp(minusyi*y_hata[i])) - linearity*yi*y_hata[i];
		if(  (y_hat[i]<0.5 && y[i]>=0.5) || (y_hat[i]>=0.5 && y[i]<0.5) )
			++hl;
	}
}

// This section is created because lbfgs needs function pointers 
// for functions that calculate things like losses (e.g. evaluate() ).
// These function pointers cannot point to member functions because
// there is a type mismatch. 
// Given that our program is single threaded, we can let the calling class
// set the vars that the evaluate() fn needs, as static vars temporarily.
// We can avoid this approach by wrapping functions and passing the wrapped 
// type, but this is simple and avoids debugging nightmares.

// Internal function to be used by the LBFGS library.
lbfgsfloatval_t Compatibility::evaluate(void *instance,
								const lbfgsfloatval_t *wv,
								lbfgsfloatval_t *g,
								const int n,
								const lbfgsfloatval_t step) {
	// Copy over some needed vars locally
	int p = model->p, d = model->d, k = model->k, m = model->m;
	data_t& xtr = model->xtr;
	data_t& ytr = model->ytr;
	floatnumber C = model->C;

	floatnumber curloss=0.0, loss=0.0, hl=0.0; // NLL and HammingLoss
	floatnumber wrongtags=0,wronglabels=0;
	const parameters w = parameters(p, d, k, wv); //w.init(p,d,k,wv);
	parameters jacobian(p, d, k, false);

	
	for(int i = 0; i < m; ++i)
	{
		record_t x = xtr[i]; record_t y = ytr[i];
		
		floatnumber y_hata[k], y_hat[k], ha[d], h[d];
		model->forwardPropagate(x,w,y_hata, y_hat,ha,h);

		model->calculateJacobian(x,y,y_hata,y_hat,ha,h,w,jacobian);
		model->calculateLosses(y_hata,y_hat,y,w, curloss, hl);

		loss += curloss;
		wrongtags += hl;
		if(hl>0) ++wronglabels;
	}
	
	// Add regularization
	floatnumber wnorm = 0;
	for(int i=0; i<n; ++i)
	{
		g[i] = C*jacobian[i] + 2*m*wv[i];
		wnorm += pow(wv[i],2);
	}
	jacobian.destroy();
	loss = C*loss + m*wnorm;
	return loss;
}

// Anything on the heap is destroyed here
singleLayerNN::~singleLayerNN() {
	 if (wopt) {
	 	wopt->destroy();
	 	delete wopt;
	 }
}

// Internal function to be used by the LBFGS library. 
// The return value of this function outputs the progress made so far at periodic intervals.
int Compatibility::progress(void *instance,
				    const lbfgsfloatval_t *x, 
				    const lbfgsfloatval_t *g, 
				    const lbfgsfloatval_t fx, 
				    const lbfgsfloatval_t xnorm,
				    const lbfgsfloatval_t gnorm,
				    const lbfgsfloatval_t step,
				    int n,
				    int k,
				    int ls) {
	int & counter = model->counter;
	++counter;
	if(counter % 1000 != 0)
		return 0;
    printf("\t\tIteration %d | ", counter); 
    printf("loss: %f; ", fx);
    printf("w-norm = %f, jacobian-norm = %f, step = %f\n", xnorm, gnorm, step);
    return 0;
}

singleLayerNN * Compatibility::model;