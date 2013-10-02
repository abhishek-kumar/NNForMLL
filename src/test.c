#include "nn.h"
#include "io.h"
#include "parameters.h"
#include "BN_MLL.h"


void testparameters()
{
	int p=10,d=3,k=2;
	parameters pp(p,d,k,false);
	
	int c=0;
	for(int i=0;i<p;++i)
		for(int j=0;j<d;++j)
			pp.val(0,i,j) = ++c;
	for(int j=0;j<d;++j)
		pp.val(0,p,j) = ++c;

	for(int i=0;i<d;++i)
		for(int j=0;j<k;++j)
			pp.val(1,i,j) = ++c;
	for(int j=0;j<k;++j)
		pp.val(1,d,j) = ++c;
	
	for(int i=0;i<p;++i)
		for(int j=0;j<k;++j)
			pp.val(2,i,j) = ++c;
	for(int j=0;j<k;++j)
		pp.val(2,p,j) = ++c;
	
	// print
	cout << "The following values should in increasing order, beginning 1\n";
	for(int i=0;i<pp.N;++i) cout << " " << pp.parametervector[i];

	cout << "\n\nThe following should all be 0\n";
	parameters ppp(p,d,k,false);
	for(int i=0;i<ppp.N;++i) cout << " " << ppp.parametervector[i];
	cout << endl;

}
/*
static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    lbfgsfloatval_t fx = 0.0;

		fx = pow(x[0]-10,2);
		g[0] = 2*(x[0]-10);

    return fx;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f\n", fx, x[0]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}
*/
void testlbfgs()
{
	int p=3,d=2,k=2;
	io fileio; fileio.readTrainingData((char *)"t.csv",p,k);
	fileio.normalize(p);
	cout << "Read Training Data" << endl;
	for(data_t::iterator it = fileio.xtr.begin(); it!=fileio.xtr.end(); ++it)
	{
		cout << "\t";
		for(record_t::iterator rit = it->begin(); rit != it->end(); ++rit)
			cout << *rit << " ";
		cout << endl;
	}
	BN_MLL model(fileio.xtr, fileio.ytr);
	model.fit(p,d,k,256);
}

int main(int argc, char **argv)
{
	cout << "Running tests...\n";
	cout << "\tParameters test: \n\t";
	testparameters();

	cout << "LBFGS test\n";
	testlbfgs();
}
