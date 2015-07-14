function gmm() {
	var math = mathematics(),
		n,
		dims,
		k,
		gaussians = [];

	model.train = function(data, options) {
		options = options || {},
		k = options["k"] || 4,
		n = data.length,
		dims = data[0].length;

		var km = kmeans(),
			means = km.train(data, options).centroids(),
			clusters = km.clusters(),
			responsibilities = math.zeroMatrix(n, k),
			maxIter = options["maxIter"] || 500,
			sum, totalRes;

		// Initialize responsibilities to either 0 or 1
		for (var i in data) {
			for (var c = 0; c < k; c++) {
				responsibilities[i][c] = (km.predict(data[i]) == c) ? 1 : 0;
			}
		}

		// Create gaussians from centroids from kmeans
		for (var i in means) {
			var weight = clusters[i].length / n;
			gaussians.push(gaussian().init(i, means[i], responsibilities, n, data));
		}

		for (var iter = 0; iter < maxIter; iter++) {
			console.log(iter)

			// Calculate responsibilities
			for (var i in data) {
				sum = 0;

				for (var c = 0; c < k; c++) { 
					sum += gaussians[c].pdf(data[i]) * gaussians[c].weight();
				}

				for (var c = 0; c < k; c++) {
					responsibilities[i][c] = (gaussians[c].pdf(data[i]) * gaussians[c].weight()) / sum;
				}
			}

			// Recompute weights, means, and covariances
			for (var c = 0; c < k; c++) {
				var totalRes = 0;
				for (var i in data) {
					 totalRes += responsibilities[i][c];
				}

				gaussians[c].weight(totalRes / n);

				for (var d = 0; d < dims; d++) {
					sum = 0;
					for (var i in data) {
						sum += (responsibilities[i][c] * data[i][d]) / totalRes;
					}

					gaussians[c].dim(d, sum);
				}

				gaussians[c].updateCovariances(responsibilities, data);
			}
		}

		return model;
	}

	model.predict = function(point) {
		var max = 0,
			id = -1;

		for (var i in gaussians) {
			if (gaussians[i].pdf(point) > max) {
				max = gaussians[i].pdf(point);
				id = i;
			}
		}

		return [i, max];
	}

	function model() {}
	return model;
}

function gaussian() {
	var math = mathematics(),
		mean,
		responsibilities = 0,
		covariances,
		weight,
		data,
		n,
		ID,
		dims;

	gauss.init = function(id, point, resp, size, points) {
		dims = point.length,
		mean = point,
		data = points,
		ID = id,
		n = size,
		covariances = math.zeroMatrix(dims, dims),
		weight = 1;

		for (var i = 0; i < n; i++) {
			responsibilities += resp[i][ID];
		}

		var sum = 0;
		for (var i = 0; i < dims; i++) {
			for (var j = 0; j < dims; j++) {
				sum = 0;

				for (var d = 0; d < n; d++) {
					sum += resp[d][ID] * (data[d][i] - mean[i]) * (data[d][j] - mean[j]);
				}

				covariances[i][j] = sum / responsibilities;
			}
		}

		return gauss;
	}

	gauss.pdf = function(point) {
		var det = math.determinant(covariances),
			left = 1 / Math.sqrt(Math.pow(2 * Math.PI, dims) * det),
			sub = math.subtractVectors(point, mean),
			inside = (-.5 * math.dotProduct(math.outerProduct(math.inverse(covariances), sub), sub));

		return left * Math.exp(inside);
	}

	gauss.dim = function(d, val) {
		if (!val) return mean[d];

		mean[d] = val;
		return gauss;
	}

	gauss.weight = function(val) {
		if (!arguments.length) return weight;
		weight = val;
		return gauss;
	}

	gauss.updateCovariances = function(resp) {
		var sum = 0,
			total = 0;

		for (var d = 0; d < n; d++) {
			total += resp[d][ID];
		}

		for (var i = 0; i < dims; i++) {
			for (var j = 0; j < dims; j++) {
				sum = 0;

				for (var d = 0; d < n; d++) {
					sum += resp[d][ID] * (data[d][i] - mean[i]) * (data[d][j] - mean[j]);
				}

				covariances[i][j] = sum / total;
			}
		}
	}

	function gauss() {	}
	return gauss;
}