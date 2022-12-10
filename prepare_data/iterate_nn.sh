set -e

queries=../keys.txt
bank=~/public/datastore/yelp_bank/large_yelp_bank.txt

for k in  100 200 300 400 500 600 700 800 900 1000; do
    python ./get_nearest_neighbors.py  --keys $queries --bank $bank --k $k --output '~/public/datastore/yelp_bank/nearest_neighbors_k'$k'.csv'
done

