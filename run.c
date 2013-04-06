#include "nn.h"
#include "io.h"
#include "parameters.h"
#include "singleLayerNN.h"
#include "fullyconnectedNN.h"
#include "BRSingleLayerNN.h"

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
		
		//bestC = trainFullyConnectedNN(fileio, p, d, k, trainfile, singleLayerC);

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
