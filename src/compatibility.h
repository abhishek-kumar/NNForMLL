#ifndef COMPATIBILITY_H
#define COMPATIBILITY_H

#include "types.h"

class BN_MLL;

// This section is created because lbfgs needs function pointers
// for functions that calculate things like losses (e.g. evaluate() ).
// These function pointers cannot point to member functions because
// there is a type mismatch (liblbfgs is pure C based, whereas we want to use C++).
// Given that our program is single threaded, we can let the calling class
// set the vars that the evaluate() fn needs, as static vars temporarily.
// We can avoid this approach by wrapping functions and passing the wrapped
// type, but this is simple and avoids debugging nightmares.
namespace Compatibility {

  extern BN_MLL * model;

  lbfgsfloatval_t evaluate(
      void *instance,
      const lbfgsfloatval_t *wv,
      lbfgsfloatval_t *g,
      const int n,
      const lbfgsfloatval_t step);

  int progress(
      void *instance,
      const lbfgsfloatval_t *x,
      const lbfgsfloatval_t *g,
      const lbfgsfloatval_t fx,
      const lbfgsfloatval_t xnorm,
      const lbfgsfloatval_t gnorm,
      const lbfgsfloatval_t step,
      int n,
      int k,
      int ls);
};

#endif
