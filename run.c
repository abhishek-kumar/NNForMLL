#include "nn.h"
#include "io.h"
#include "parameters.h"
#include "singleLayerNN.h"
#include "fullyconnectedNN.h"
#include "BRSingleLayerNN.h"

/* 
 * Trains a regularized single layer model and returns the bestC
 
singleLayerNN * trainSingleLayerNN(io & fileio, int p, int d, int k)
{
	// cross validate to find regularization strength
	int m = fileio.xtr.size(), folds = 5;
	int indexes[m]; for(int i=0; i<m; ++i) indexes[i] = i; randomShuffle(indexes, m); 
	
	floatnumber bestC=0.0, bestNLL=1e10, lastNLL=1e10, lasttolastNLL=1e10;
	floatnumber bestNllMean, bestNllStdev;
	floatnumber bestHlMean, bestHlStdev, bestSlMean, bestSlStdev, bestRlMean, bestRlStdev, bestNrlMean, 
			bestNrlStdev, bestOeMean, bestOeStdev, bestAvprecMean, bestAvprecStdev;
	//for(int cc=-12; cc<13; cc+=1)
	for(int cc=6; cc<13; cc+=1)
	{
		floatnumber C = pow(2,cc);
		record_t cvLossesNll, cvLossesHl, cvLossesSl, cvLossesRl, cvLossesNrl, cvLossesOe, cvLossesAvprec;
		for(int cv=0; cv<folds; ++cv)
		{
			int testsetstart = cv*float(m)/folds;
			int testsetend = testsetstart+float(m)/folds;
			data_t cvxtrain, cvytrain; cvxtrain.clear(); cvytrain.clear();
			data_t cvxtest, cvytest; cvxtest.clear(); cvytest.clear();
			for(int i=0; i<testsetstart; ++i)
			{
				cvxtrain.push_back(fileio.xtr[indexes[i]]);
				cvytrain.push_back(fileio.ytr[indexes[i]]);
			}
			for(int i=testsetstart; i<testsetend; ++i)
			{
				cvxtest.push_back(fileio.xtr[indexes[i]]);
				cvytest.push_back(fileio.ytr[indexes[i]]);
			}
			for(int i=testsetend; i<m; ++i)
			{
				cvxtrain.push_back(fileio.xtr[indexes[i]]);
				cvytrain.push_back(fileio.ytr[indexes[i]]);
			}

			singleLayerNN tempModel(cvxtrain, cvytrain);
			tempModel.fit(p,d,k,C);
			floatnumber nll, sl, hl, rl, nrl, oe, avprec; 
			tempModel.test(cvxtest, cvytest, nll, sl, hl, rl, nrl, oe, avprec);
			cvLossesNll.push_back(nll); cvLossesHl.push_back(hl); cvLossesSl.push_back(sl); 
			cvLossesRl.push_back(rl); cvLossesNrl.push_back(nrl); cvLossesOe.push_back(oe); cvLossesAvprec.push_back(avprec);
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
	     << bestC << " || NLL=" << bestNLL << endl;
	singleLayerNN * BNMLL = new singleLayerNN(fileio.xtr, fileio.ytr);
	BNMLL->fit(p,d,k,bestC);

	ofstream resultsfile; resultsfile.open("resultstrain_sl.txt", ios::app);
	resultsfile << endl << "Results for [" << fileio.trainFileName << "]" << endl;
	resultsfile << d << "\t" << bestC << "\t" << bestNllMean << " +- " << bestNllStdev << "\t" 
	            << bestHlMean << " +- " << bestHlStdev << "\t"
	            << bestSlMean << " +- " << bestSlStdev << "\t"
	            << bestRlMean << " +- " << bestRlStdev << "\t"
	            << bestNrlMean << " +- " << bestNrlStdev << "\t"
	            << bestOeMean << " +- " << bestOeStdev << "\t"
	            << bestAvprecMean << "\t" << bestAvprecStdev << endl;
	resultsfile.close();

	return BNMLL;
}
*/

// Trains a regularized fully connected model and returns the bestC
floatnumber trainFullyConnectedNN(io & fileio, int p, int d, int k, char *trainfile, floatnumber singleLayerC)
{
	// First train the single layer model to obtain initial parameter estimates
	singleLayerNN BNMLL(fileio.xtr, fileio.ytr, p, d, k, singleLayerC);
	BNMLL.fit();

	// cross validate to find regularization strength
	int m = fileio.xtr.size(), folds = 5;
	int indexes[m]; for(int i=0; i<m; ++i) indexes[i] = i; randomShuffle(indexes, m); 
	
	floatnumber bestC=0.0, bestNLL=1e10, lastNLL=1e10, lasttolastNLL=1e10;
	floatnumber bestNllMean, bestNllStdev;
	floatnumber bestHlMean, bestHlStdev, bestSlMean, bestSlStdev, bestRlMean, bestRlStdev, bestNrlMean, 
			bestNrlStdev, bestOeMean, bestOeStdev, bestAvprecMean, bestAvprecStdev;
	for(int cc=-4; cc<13; cc+=2)
	{
		floatnumber C = pow(2,cc); 
		record_t cvLossesNll, cvLossesHl, cvLossesSl, cvLossesRl, cvLossesNrl, cvLossesOe, cvLossesAvprec;
		for(int cv = 0; cv < folds; ++cv)
		{
			int testsetstart = cv*float(m)/folds;
			int testsetend = testsetstart+float(m)/folds;
			data_t cvxtrain, cvytrain; cvxtrain.clear(); cvytrain.clear();
			data_t cvxtest, cvytest; cvxtest.clear(); cvytest.clear();
			for(int i=0; i<testsetstart; ++i)
			{
				cvxtrain.push_back(fileio.xtr[indexes[i]]);
				cvytrain.push_back(fileio.ytr[indexes[i]]);
			}
			for(int i=testsetstart; i<testsetend; ++i)
			{
				cvxtest.push_back(fileio.xtr[indexes[i]]);
				cvytest.push_back(fileio.ytr[indexes[i]]);
			}
			for(int i=testsetend; i<m; ++i)
			{
				cvxtrain.push_back(fileio.xtr[indexes[i]]);
				cvytrain.push_back(fileio.ytr[indexes[i]]);
			}

			fullyConnectedNN::fit(cvxtrain, cvytrain, p,d,k,singleLayerC,BNMLL.getParameters(), C);
			floatnumber nll, sl, hl, rl, nrl, oe, avprec; 
			fullyConnectedNN::test(cvxtest, cvytest, nll, sl, hl, rl, nrl, oe, avprec);
			cvLossesNll.push_back(nll); cvLossesHl.push_back(hl); cvLossesSl.push_back(sl); 
			cvLossesRl.push_back(rl); cvLossesNrl.push_back(nrl); cvLossesOe.push_back(oe); cvLossesAvprec.push_back(avprec);
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
	cout << "Regularization Parameter Training | Best C: " 
	     << bestC << " || NLL=" << bestNLL << endl;
	fullyConnectedNN::fit(fileio.xtr, fileio.ytr, p,d,k,singleLayerC, BNMLL.getParameters(), bestC);

	ofstream resultsfile; resultsfile.open("resultstrain_fc.txt", ios::app);
	resultsfile << endl << "Results for [" << trainfile << "]" << endl;
	resultsfile << d << "\t" << bestC << "\t" << bestNllMean << " +- " << bestNllStdev << "\t" 
	            << bestHlMean << " +- " << bestHlStdev << "\t"
	            << bestSlMean << " +- " << bestSlStdev << "\t"
	            << bestRlMean << " +- " << bestRlStdev << "\t"
	            << bestNrlMean << " +- " << bestNrlStdev << "\t"
	            << bestOeMean << " +- " << bestOeStdev << "\t"
	            << bestAvprecMean << "\t" << bestAvprecStdev << endl;
	resultsfile.close();

	return bestC;
}

int main(int argc, char **argv)
{
	if(argc<7)
	{
		cout << "Usage: mll <1|2|3> <train file> <test file> <p> <h> <k> [singleLayerC]" << endl;
		return 1;
	}
	srand ( time(NULL) );
	char *trainfile = argv[2]; char *testfile = argv[3];
	int p = atoi(argv[4]), d=atoi(argv[5]), k=atoi(argv[6]);
	io fileio; fileio.readTrainingData(trainfile,p,k); fileio.readTestData(testfile,p,k);
	fileio.normalize(p);
	floatnumber singleLayerC = -1.0;
	

	floatnumber bestC; //, nll, sl, hl, rl, nrl, oe, avprec; 
	error_t loss;
	ofstream resultsfile;

	if(atoi(argv[1]) == 1)
	{
		singleLayerNN BNMLL(fileio, p, d, k);
		BNMLL.train(-12, 13, 2);
		bestC = BNMLL.getRegularizationStrength();
		loss = BNMLL.test(fileio.xte, fileio.yte);

		// Print out some interesting tag-correlations
		printTagCorrelations(*(BNMLL.getParameters()), 6);
		resultsfile.open("resultstest_sl.txt", ios::app);
	} 
	else if(atoi(argv[1]) == 2)
	{
		if(argc < 8) { cout << "You must provide 7 arguments, the last one being C" << endl; return 1;	 }
		singleLayerC = atof(argv[7]);
		
		bestC = trainFullyConnectedNN(fileio, p, d, k, trainfile, singleLayerC);

		//singleLayerNN::test(fileio.xte, fileio.yte, nll, sl, hl, rl, nrl, oe, avprec);
		//cout << "Performance of the single layer NN:\n";
		//cout << "\tNegative Log Likelihood: " << nll << endl;
		//cout << "\t0/1 Subset Loss: " << sl << endl;
		//cout << "\tHamming Loss: " << hl << endl;
		//cout << "\tRanking Loss: " << rl << endl;
		//cout << "\tNormalized Ranking Loss: " << nrl << endl;
		//resultsfile.open("resultstest_sl.txt", ios::app);
		//resultsfile << endl << "Results for [" << trainfile << "]" << endl;
		//resultsfile << d << "\t" << bestC << "\t" << nll << "\t" << sl << "\t"
	  	//          << hl << "\t" << rl << "\t" << nrl << "\t" << oe << "\t" << avprec << endl;
		//resultsfile.close();

		//fullyConnectedNN::test(fileio.xte, fileio.yte, nll, sl, hl, rl, nrl, oe, avprec);
		//resultsfile.open("resultstest_fc.txt", ios::app);
		throw "Unsupported Operation";
	} if(atoi(argv[1]) == 3) {
		BRSingleLayerNN model(fileio, p, d, k);
		model.train(-12, 13, 2);
		loss = model.test(fileio.xte, fileio.yte);
		resultsfile.open("resultstest_brsl.txt", ios::app);
	}
	else
	{
		cerr << "Error, unknown first argument provide. Must be either 1 (BN-MLL), 2 (SLN-MLL) or 3 (BR-BNMLL)." << endl;
		return 1;
	}
	cout << "Performance on Test Set:\n";
	loss.print();

	resultsfile << endl << "Results for [" << trainfile << "]" << endl;
	resultsfile << d << "\t[" << singleLayerC << "|" << bestC << "]\t" << loss.nll << "\t" << loss.sl << "\t"
	            << loss.hl << "\t" << loss.rl << "\t" << loss.nrl << "\t" << loss.oe << "\t" << loss.avprec << endl;
	resultsfile.close();
}
