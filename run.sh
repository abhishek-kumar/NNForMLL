#!/bin/bash

# Check if our binary has been created
if [ ! -f "./bin/mll" ]; then
  echo "Binary not found. Please run make before running this script."
  echo "  $ make "
  exit 1
fi

cd data
datasets=$(ls -d */ | cut -f1 -d '/')
cd ..
if [ -z "$datasets" ]; then
  echo "Datasets not found. Please run make within the data directory before"
  echo "running this script"
  echo "  $ cd data"
  echo "  $ make"
  exit 1
fi

echo
echo "We will train on 3 benchmark datasets and report test set results"
echo "In this run, regularization strength will be fixed"
echo "(not trained via Cross validation)"
echo
rm -rf mll.log

for dataset in $datasets; do
  if [ "$dataset" == "nn-test" ]; then
    # Ignore the test dataset, which is only for the unit test.
    continue
  fi

  # Read dataset dimensions
  p=$(cat data/$dataset/dimensions.txt | grep 'p=' | cut -f2 -d '=')
  k=$(cat data/$dataset/dimensions.txt | grep 'k=' | cut -f2 -d '=')
  h=$(cat data/$dataset/dimensions.txt | grep 'h=' | cut -f2 -d '=')
  C=$(cat data/$dataset/dimensions.txt | grep 'C=' | cut -f2 -d '=')

  # Train and test
  echo "  Processing dataset '$dataset' (p = $p, k = $k, h = $h)"
  echo -n -e "  To check ongoing progress, see mll.log\r"
  if [ -f "/usr/bin/time" ]; then
    /usr/bin/time \
      bin/mll 1 data/$dataset/$dataset-train.csv \
      data/$dataset/$dataset-test.csv  $p $h $k $C 2>&1 \
      | sed -e "s/ *\(.*\)real.*/    Time taken: \1 seconds./" \
      | sed -e "s/\(.*\)user.*/    Time taken: \1 seconds./" \
      | grep -v "pagefaults"
  else
    bin/mll 1 data/$dataset/$dataset-train.csv \
      data/$dataset/$dataset-test.csv  $p $h $k $C
  fi
  # Note to anyone reading: If you want to train the regularization weight, then
  # don't provide the last argument $C. The program will cross validate and
  # learn the best value.

  mv mll.log $dataset.log
  echo "    See $dataset.log for more details, including label and feature correlations"
done
