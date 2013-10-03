#include "types.h"

#define PI 3.141592654

/*
  Gaussian Distribution sampling
  Courtesy http://c-faq.com/lib/gaussian.html
  I've chosen an implementation that is not the fastest, but is more accurate
*/
double gaussrand();

typedef struct sparameters{
  floatnumber * parametervector;
  const floatnumber * cparametervector;
  int p,d,k,layer1N,layer2N,layer3N,N;

  // p,d,k are self explanatory
  // initializevector = 1 means vector will be initialized randomly
  //                      with appropriate gaussian
  // initializevector = 0 meanse vector will set to all zero
  sparameters(int pp, int dd, int kk, bool initializevector = false);

  // Initialize from a previous set of values
  sparameters(int pp, int dd, int kk, const floatnumber *w);

  floatnumber & val(int layer, int inputidx, int outputidx);

  floatnumber val(int layer, int inputidx, int outputidx) const;

  floatnumber & operator[](const int idx);

  floatnumber operator[](const int idx) const;

  floatnumber * getvector();

  const floatnumber * getvector() const;

  // Should be called by the caller of init()
  // Except when initialized from previous set of values
  void destroy();

} parameters;

bool comparator ( const std::pair<floatnumber, int>& l, const std::pair<floatnumber, int>& r);

/*
 * Prints the top tags for each hidden node
 * p: parameter structure
 * n: Number of tags to print for each hidden node
 */
void LogTagCorrelations(parameters& p, int n);
