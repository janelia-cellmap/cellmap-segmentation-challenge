// Code originally from funlib.evaluate@git+//github.com/funkelab/funlib.evaluate
#ifndef IMPL_RAND_VOI_H__
#define IMPL_RAND_VOI_H__

#include <map>
#include <math.h>
#include <cmath>

struct Metrics {

	double voi_split;
	double voi_merge;
	double rand_split;
	double rand_merge;
	double nvi_split;
	double nvi_merge;
	double nid;

	std::map<uint64_t, double> voi_split_i;
	std::map<uint64_t, double> voi_merge_j;
};

template <typename V1, typename V2>
Metrics
rand_voi_arrays(
		std::size_t size,
		const V1* labels_a,
		const V2* labels_b,
		bool return_cluster_scores=false){

	double total = 0;

	// number of co-occurences of label i and j
	std::map<uint64_t, std::map<uint64_t, double>> p_ij;

	// number of occurences of label i and j in the respective volumes
	std::map<uint64_t, double> p_i, p_j;

	for (std::ptrdiff_t i = 0; i < size; ++i) {

		uint64_t a = labels_a[i];
		uint64_t b = labels_b[i];

		if (a) {

			++total;

			++p_ij[a][b];
			++p_i[a];
			++p_j[b];
		}
	}

	// sum of squares in p_ij
	double sum_p_ij = 0;
	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			sum_p_ij += b.second * b.second;

	// sum of squares in p_i
	double sum_p_i = 0;
	for ( auto& a: p_i )
		sum_p_i += a.second * a.second;

	// sum of squares in p_j
	double sum_p_j = 0;
	for ( auto& b: p_j )
		sum_p_j += b.second * b.second;

	// we have everything we need for RAND, normalize histograms for VOI

	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			b.second /= total;

	for ( auto& a: p_i )
		a.second /= total;

	for ( auto& b: p_j )
		b.second /= total;

	// compute entropies

	std::map<uint64_t, double> voi_split_i;
	std::map<uint64_t, double> voi_merge_j;

	if (return_cluster_scores) {

		for ( auto& a: p_i )
			voi_split_i[a.first] = a.second * log2(a.second);
		for ( auto& b: p_j )
			voi_merge_j[b.first] = b.second * log2(b.second);
	}

	// H(a,b)
	double H_ab = 0;
	for ( auto& a: p_ij )
		for ( auto& b: a.second ) {

			if(b.second) {

				H_ab -= b.second * log2(b.second);

				if (return_cluster_scores) {

					voi_split_i[a.first] -= b.second * log2(b.second);
					voi_merge_j[b.first] -= b.second * log2(b.second);
				}
			}
		}

	// H(a)
	double H_a = 0;
	for ( auto& a: p_i )
		if(a.second)
			H_a -= a.second * log2(a.second);

	// H(b)
	double H_b = 0;
	for ( auto& b: p_j )
		if(b.second)
			H_b -= b.second * log2(b.second);

	double rand_split = sum_p_ij/sum_p_i;
	double rand_merge = sum_p_ij/sum_p_j;

	// H(b|a)
	double voi_split = H_ab - H_a;
	// H(a|b)
	double voi_merge = H_ab - H_b;
	
	// normalized measures
	double nvi_split = voi_split/H_ab;
	double nvi_merge = voi_merge/H_ab;
	double nid = 1 - ((H_a + H_b - H_ab)/fmax(H_a,H_b));

	Metrics metrics;
	metrics.rand_split = rand_split;
	metrics.rand_merge = rand_merge;
	metrics.voi_split  = voi_split;
	metrics.voi_merge  = voi_merge;
	metrics.nvi_split = nvi_split;
	metrics.nvi_merge = nvi_merge;
	metrics.nid = nid;
	metrics.voi_split_i = voi_split_i;
	metrics.voi_merge_j = voi_merge_j;

	return metrics;
}

#endif // IMPL_RAND_VOI_H__
