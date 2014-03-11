#include "parameters.h"

#include "logging.h"
#include "types.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
  Gaussian Distribution sampling
  Courtesy http://c-faq.com/lib/gaussian.html
  I've chosen an implementation that is not the fastest, but is more accurate
*/
double GaussRand() {
  static double V1, V2, S;
  static int phase = 0;
  double X;

  if(phase == 0) {
    do {
      double U1 = (double)rand() / RAND_MAX;
      double U2 = (double)rand() / RAND_MAX;

      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
      } while(S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);
  } else
    X = V2 * sqrt(-2 * log(S) / S);

  phase = 1 - phase;

  return X;
}

// p,d,k are self explanatory
// initializevector = 1 means vector will be initialized randomly
//                      with appropriate gaussian
// initializevector = 0 means vector will set to all zero
sparameters::sparameters(
    int p_, int d_, int k_,
    bool initialize_layer1, bool initialize_layer2, bool initialize_layer3) {
  p = p_; d=d_; k = k_;
  layer1N=(p+1)*(d); layer2N=(d+1)*k; layer3N=(p+1)*k;
  N = layer1N + layer2N + layer3N;
  parametervector = lbfgs_malloc(N);
  cparametervector = parametervector; // const pointer to same data
  floatnumber sigma = 1.0; int i; floatnumber *f;

  // Initialize all weights to 0
  for (i=0, f=parametervector; i<N; ++i,++f)
    *f = 0.0;

  // Initialize the weights from a Gaussian distribution with
  // mean = 0 and std deviation = 1 / sqrt(number of units)
  if (initialize_layer1) {
    sigma = 1/sqrt(p+1);
    for(i=0, f=parametervector; i<layer1N; ++i,++f)
      *f = sigma*GaussRand();
  }

  if (initialize_layer2) {
    sigma = 1/sqrt(d+1);
    for(i=0, f=parametervector+layer1N; i<layer2N; ++i,++f)
      *f = sigma*GaussRand();
  }

  if (initialize_layer3) {
    sigma = 1/sqrt(p+1);
    for(i=0, f=parametervector+layer1N+layer2N; i<layer3N; ++i,++f)
      *f = sigma*GaussRand();
  }
}

// Initialize from a previous set of values
sparameters::sparameters(int pp, int dd, int kk, const floatnumber *w) :
    parametervector(NULL), cparametervector(w),
    p(pp), d(dd), k(kk),
    layer1N((pp+1)*(dd)), layer2N((dd+1)*kk), layer3N((pp+1)*kk), N(0.0) {
  N = layer1N + layer2N + layer3N;
}

floatnumber & sparameters::val(int layer, int inputidx, int outputidx)
{
  if(layer==0)
    return *(parametervector+inputidx*(d)+outputidx);
  else if(layer==1)
    return *(parametervector+layer1N+inputidx*(k)+outputidx);
  else if(layer==2)
    return *(parametervector+layer1N+layer2N+inputidx*(k)+outputidx);
  else
    throw "wrong parameter in 'layer'";
}

floatnumber sparameters::val(int layer, int inputidx, int outputidx) const
{
  if(layer==0)
    return *(cparametervector+inputidx*(d)+outputidx);
  else if(layer==1)
    return *(cparametervector+layer1N+inputidx*(k)+outputidx);
  else if(layer==2)
    return *(cparametervector+layer1N+layer2N+inputidx*(k)+outputidx);
  else
    throw "wrong parameter in 'layer'";
}

floatnumber & sparameters::operator[](const int idx)
{
  return parametervector[idx];
}

floatnumber sparameters::operator[](const int idx) const
{
  return cparametervector[idx];
}

floatnumber * sparameters::getvector()
{
  return parametervector;
}

const floatnumber * sparameters::getvector() const
{
  return cparametervector;
}


// Should be called by the caller of init()
// Except when initialized from previous set of values
void sparameters::destroy()
{
  if (parametervector)
    lbfgs_free(parametervector);
}

bool comparator ( const std::pair<floatnumber, int>& l, const std::pair<floatnumber, int>& r)
   { return l.first < r.first; }

/*
 * Prints the top tags for each hidden node
 * p: parameter structure
 * n: Number of tags to print for each hidden node
 */
void LogTagCorrelations(parameters& p, int n) {
  // Output top n labels for each hidden unit
  Log("Correlations between labels, as seen by hidden units:");
  for(int h=0; h<(p.d+1); ++h) {
    std::vector<std::pair<floatnumber,int> > weights;
    for(int j=0; j<p.k; ++j) {
      std::pair<floatnumber, int> weight_and_index((-1.0)*fabs(p.val(1,h,j)), j);
      weights.push_back(weight_and_index);
    }
    std::sort(weights.begin(), weights.end(), comparator);
    Log("\tTop Labels for hidden node number %d:", h);
    for(int i=0;i<n;++i)
      Log("\t\tLabel #%d with weight: %f", \
          weights[i].second, p.val(1,h,weights[i].second));
  }

  // Output top n features for each hidden unit
  Log("Correlations between features, as seen by hidden units:");
  for(int h=0; h<(p.d+1); ++h) {

    std::vector<std::pair<floatnumber,int> > weights;
    for(int j=0; j<p.p; ++j) {
      std::pair<floatnumber, int> weight_and_index((-1.0)*fabs(p.val(0,j,h)), j);
      weights.push_back(weight_and_index);
    }
    std::sort(weights.begin(), weights.end(), comparator);
    Log("\tTop Features for hidden node number %d:", h);
    for(int i=0;i<n;++i)
      Log("\t\tFeature #%d with weight: %f", \
          weights[i].second, p.val(0, weights[i].second, h));
  }
}
