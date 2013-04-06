// Structure that holds a dataset, including training set, test set, training labels, and test labels. 
typedef struct
{
	// Training set, test set, training labels, test labels (ground truth)
	data_t xtr, xte, ytr, yte;
	floatnumber * xtrmean; floatnumber * xtrrmse;
	string trainFileName, testFileName;

	 // Read training data with p features and k tags in its label, from file.
	void readTrainingData(char *filename, int p, int k)
	{
		ifstream infile(filename);
		trainFileName = string(filename);
		xtr.clear(); ytr.clear();
		xtrmean = (floatnumber *) malloc(sizeof(floatnumber)*p);
		xtrrmse = (floatnumber *) malloc(sizeof(floatnumber)*p);
		for(int i=0; i<p; ++i)
		{
			xtrmean[i] = 0.0;
			xtrrmse[i] = 0.0;
		}

		while(infile)
		{
			string line; if (!getline(infile, line)) break;
			istringstream sline(line);
			record_t xrecord, yrecord;

			//Read input features (x_i)
			for(int i=0; i<p; ++i)
			{
				string field; if (!getline(sline, field, ',')) throw "Unexpected end of file";
				floatnumber f = atof(field.c_str());
				xrecord.push_back(f);
				xtrmean[i] += f;
			}

			//Read the output tags (y_i)
			for(int i=0; i<k; ++i)
			{
				string field; if (!getline(sline, field, ',')) throw "Unexpected end of file";
				yrecord.push_back(atof(field.c_str()));
			}

			xtr.push_back(xrecord); 
			ytr.push_back(yrecord);
		};
	}

	/**
	 * Normalize the dataset to have 0-mean and 1-variance.
	 */
	void normalize(int p)
	{

		// Calculate mean
		floatnumber m = (floatnumber) xtr.size();
		for(int i=0; i<p; ++i) xtrmean[i] /= m;

		// Calculate std deviation
		for(data_t::iterator it = xtr.begin(); it!=xtr.end(); ++it)
			for(int i=0; i<p; ++i)
				xtrrmse[i] += pow((*it)[i]-xtrmean[i],2);
		for(int i=0; i<p; ++i) 
			xtrrmse[i] = sqrt(xtrrmse[i] / m);

		// Update data to 0-mean 1-variance
		// Training Set
		for(data_t::iterator it = xtr.begin(); it!=xtr.end(); ++it)
			for(int i=0; i<p; ++i)
				(*it)[i] = ((*it)[i]-xtrmean[i])/xtrrmse[i];

		// Test Set
		for(data_t::iterator it = xte.begin(); it!=xte.end(); ++it)
			for(int i=0; i<p; ++i)
				(*it)[i] = ((*it)[i]-xtrmean[i])/xtrrmse[i];

	}


	void readTestData(char *filename, int p, int k)
	{
		ifstream infile(filename);
		testFileName = string(filename);
		xte.clear(); yte.clear();

		while(infile)
		{
			string line; if (!getline(infile, line)) break;
			istringstream sline(line);
			record_t xrecord, yrecord;

			//Read input features (x_i)
			for(int i=0; i<p; ++i)
			{
				string field; if (!getline(sline, field, ',')) throw "Unexpected end of file";
				floatnumber f = atof(field.c_str());
				xrecord.push_back(f);
			}

			//Read the output tags (y_i)
			for(int i=0; i<k; ++i)
			{
				string field; if (!getline(sline, field, ',')) throw "Unexpected end of file";
				yrecord.push_back(atof(field.c_str()));
			}

			xte.push_back(xrecord); yte.push_back(yrecord);
		}
	}

} io;
